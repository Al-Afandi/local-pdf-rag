"""Run the local generator over a question CSV and save reviewable outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import rag


RETRIEVAL_MODES = ("dense", "hybrid", "rerank", "hybrid-rerank")
EVIDENCE_SOURCES = ("retrieved-evidence", "curated-evidence")
PROMPT_VERSION = "rag._call_ollama.v1"
FIELDNAMES = [
    "question_id",
    "question",
    "document",
    "answerability",
    "evidence_source",
    "retrieval_mode",
    "top_k",
    "context_chunk_count",
    "model",
    "embedding_model",
    "chunk_target_words",
    "chunk_max_words",
    "chunk_overlap_words",
    "temperature",
    "prompt_version",
    "context_chunk_ids",
    "context_chunks_json",
    "answer",
    "cited_source_labels",
    "cited_chunk_ids",
    "elapsed_seconds",
    "error",
    "scope_fit",
    "factual_correctness",
    "answer_completeness",
    "citation_support_overall",
    "citation_completeness",
    "abstention_correctness",
    "review_notes",
]
CITATION_FIELDNAMES = [
    "question_id", "question", "document", "evidence_source", "retrieval_mode", "top_k", "model",
    "answer", "source_label", "chunk_id", "source_document", "page_number",
    "title", "passage", "source_found", "claim_or_sentence", "support_rating",
    "relevance_rating", "citation_notes",
]


def load_questions(
    path: Path, evidence_source: str
) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"Question CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if "question" not in fields:
            raise ValueError("Question CSV must contain a 'question' column.")
        if not ({"document", "document_filter"} & fields):
            raise ValueError(
                "Question CSV must contain a 'document' or 'document_filter' column."
            )
        if evidence_source == "curated-evidence" and "relevant_chunk_ids" not in fields:
            raise ValueError(
                "Curated evidence requires a 'relevant_chunk_ids' column in the benchmark."
            )

        questions: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()
            if not question:
                raise ValueError(f"Row {row_number} has an empty question.")

            question_id = (row.get("question_id") or "").strip() or f"q{row_number - 1:03d}"
            if question_id in seen_ids:
                raise ValueError(f"Duplicate question_id {question_id!r}.")
            seen_ids.add(question_id)
            chunk_ids = list(
                dict.fromkeys(
                    item.strip()
                    for item in re.split(r"[;|]", row.get("relevant_chunk_ids") or "")
                    if item.strip()
                )
            )
            answerability = (
                (row.get("answerability") or row.get("source_answerability") or "")
                .strip()
                .casefold()
            )
            if evidence_source == "curated-evidence":
                if answerability in {"answerable", "partially_answerable"} and not chunk_ids:
                    raise ValueError(
                        f"{question_id}: answerable question has no relevant_chunk_ids."
                    )
                if answerability in {"unanswerable", "uncertain"} and chunk_ids:
                    raise ValueError(
                        f"{question_id}: {answerability} question should not list relevant_chunk_ids."
                    )

            questions.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "document": (
                        (row.get("document") or row.get("document_filter") or "").strip()
                    ),
                    "answerability": answerability,
                    "relevant_chunk_ids": chunk_ids,
                }
            )

    if not questions:
        raise ValueError("Question CSV has a header but no question rows.")
    return questions


def serialize_hits(hits: list[rag.SearchHit], evidence_source: str) -> str:
    """Keep the exact context used for generation alongside each answer."""
    return json.dumps(
        [
            {
                "rank": hit.rank,
                "score": (
                    hit.score if evidence_source == "retrieved-evidence" else None
                ),
                "chunk_id": hit.chunk_id,
                "document": hit.file_name,
                "page_number": hit.page_number,
                "title": hit.section,
                "passage": hit.text,
            }
            for hit in hits
        ],
        ensure_ascii=False,
    )


def load_curated_hits(
    chunk_ids: list[str], document_name: str | None
) -> list[rag.SearchHit]:
    """Load benchmark-labeled evidence in the exact order listed in the CSV."""
    if not chunk_ids:
        return []

    placeholders = ",".join("?" for _ in chunk_ids)
    connection = rag._connect()
    try:
        rows = connection.execute(
            "SELECT chunks.chunk_id, documents.file_name, chunks.page_number, "
            "chunks.section, chunks.text FROM chunks "
            "JOIN documents USING (document_id) "
            f"WHERE chunks.chunk_id IN ({placeholders})",
            tuple(chunk_ids),
        ).fetchall()
    finally:
        connection.close()

    by_id = {row["chunk_id"]: row for row in rows}
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in by_id]
    if missing:
        raise ValueError(
            "Benchmark chunk ID(s) are missing from the current index: "
            + ", ".join(missing)
            + ". Check the IDs or re-add the PDFs if chunking settings changed."
        )

    hits: list[rag.SearchHit] = []
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        row = by_id[chunk_id]
        if document_name and row["file_name"] != document_name:
            raise ValueError(
                f"Benchmark chunk {chunk_id!r} belongs to {row['file_name']!r}, "
                f"not the selected document {document_name!r}."
            )
        hits.append(
            rag.SearchHit(
                rank=rank,
                score=0.0,
                chunk_id=chunk_id,
                file_name=row["file_name"],
                page_number=row["page_number"],
                section=row["section"],
                text=row["text"],
            )
        )
    return hits


def evaluate(
    questions: list[dict[str, object]],
    modes: list[str],
    top_k: int,
    model: str,
    output: Path,
    overwrite: bool,
    evidence_source: str,
) -> None:
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if output.exists() and not overwrite:
        raise ValueError(
            f"Output already exists: {output}. Choose another path or use --overwrite."
        )

    citation_output = output.with_name(f"{output.stem}_citations{output.suffix}")
    if citation_output.exists() and not overwrite:
        raise ValueError(
            f"Citation review file already exists: {citation_output}. "
            "Choose another output path or use --overwrite."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    available_documents = set(rag.indexed_document_names())
    if not available_documents:
        raise ValueError("No PDFs are indexed. Add the benchmark documents first.")

    completed = 0
    failed = 0
    with output.open("w", encoding="utf-8-sig", newline="") as handle, citation_output.open(
        "w", encoding="utf-8-sig", newline=""
    ) as citation_handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        citation_writer = csv.DictWriter(
            citation_handle, fieldnames=CITATION_FIELDNAMES
        )
        writer.writeheader()
        citation_writer.writeheader()
        handle.flush()
        citation_handle.flush()

        for question in questions:
            document = str(question["document"]) or None
            run_modes = modes if evidence_source == "retrieved-evidence" else ["not_applicable"]
            if document and document not in available_documents:
                print(
                    f"{question['question_id']}: document {document!r} is not indexed; skipping.",
                    file=sys.stderr,
                )
                for mode in run_modes:
                    writer.writerow(
                        _error_row(
                            question, mode, top_k, model,
                            f"Document not indexed: {document}", evidence_source,
                        )
                    )
                    failed += 1
                handle.flush()
                continue

            if document is None and len(available_documents) > 1:
                reason = "Question has no document filter while multiple PDFs are indexed."
                for mode in run_modes:
                    writer.writerow(
                        _error_row(question, mode, top_k, model, reason, evidence_source)
                    )
                    failed += 1
                handle.flush()
                continue

            for mode in run_modes:
                print(f"{question['question_id']} | {mode} | generating...", flush=True)
                started = time.monotonic()
                try:
                    if evidence_source == "retrieved-evidence":
                        hits = rag.retrieve_by_mode(
                            str(question["question"]), top_k, document, mode
                        )
                    else:
                        hits = load_curated_hits(
                            question["relevant_chunk_ids"], document
                        )
                    if not hits and evidence_source == "retrieved-evidence":
                        raise ValueError("No chunks were retrieved for this question.")

                    answer = rag._call_ollama(str(question["question"]), hits, model)
                    labels = sorted(
                        {int(value) for value in re.findall(r"(?<!\w)\[S(\d+)\]", answer)}
                    )
                    valid_ranks = {hit.rank for hit in hits}
                    cited_ranks = set(labels)
                    cited_chunks = [
                        hit.chunk_id for hit in hits if hit.rank in cited_ranks
                    ]
                    invalid_labels = sorted(set(labels) - valid_ranks)
                    if invalid_labels:
                        print(
                            f"  Warning: invalid source labels {invalid_labels}",
                            file=sys.stderr,
                        )

                    writer.writerow(
                        {
                            "question_id": question["question_id"],
                            "question": question["question"],
                            "document": document or "",
                            "answerability": question["answerability"],
                            "evidence_source": evidence_source,
                            "retrieval_mode": (
                                mode if evidence_source == "retrieved-evidence" else "not_applicable"
                            ),
                            "top_k": top_k if evidence_source == "retrieved-evidence" else "",
                            "context_chunk_count": len(hits),
                            "model": model,
                            "embedding_model": rag.EMBEDDING_MODEL,
                            "chunk_target_words": rag.CHUNK_TARGET_WORDS,
                            "chunk_max_words": rag.CHUNK_MAX_WORDS,
                            "chunk_overlap_words": rag.CHUNK_OVERLAP_WORDS,
                            "temperature": 0,
                            "prompt_version": PROMPT_VERSION,
                            "context_chunk_ids": " | ".join(hit.chunk_id for hit in hits),
                            "context_chunks_json": serialize_hits(hits, evidence_source),
                            "answer": answer,
                            "cited_source_labels": " | ".join(f"[S{label}]" for label in labels),
                            "cited_chunk_ids": " | ".join(cited_chunks),
                            "elapsed_seconds": f"{time.monotonic() - started:.2f}",
                            "error": "",
                            "scope_fit": "",
                            "factual_correctness": "",
                            "answer_completeness": "",
                            "citation_support_overall": "",
                            "citation_completeness": "",
                            "abstention_correctness": "",
                            "review_notes": "",
                        }
                    )
                    for hit in hits:
                        if hit.rank not in cited_ranks:
                            continue
                        citation_writer.writerow(
                            {
                                "question_id": question["question_id"],
                                "question": question["question"],
                                "document": document or "",
                                "retrieval_mode": (
                                    mode if evidence_source == "retrieved-evidence" else "not_applicable"
                                ),
                                "top_k": top_k if evidence_source == "retrieved-evidence" else "",
                                "model": model,
                                "evidence_source": evidence_source,
                                "answer": answer,
                                "source_label": f"[S{hit.rank}]",
                                "chunk_id": hit.chunk_id,
                                "source_document": hit.file_name,
                                "page_number": hit.page_number,
                                "title": hit.section,
                                "passage": hit.text,
                                "source_found": "yes",
                                "claim_or_sentence": "",
                                "support_rating": "",
                                "relevance_rating": "",
                                "citation_notes": "",
                            }
                        )
                    for label in invalid_labels:
                        citation_writer.writerow(
                            {
                                "question_id": question["question_id"],
                                "question": question["question"],
                                "document": document or "",
                                "retrieval_mode": (
                                    mode if evidence_source == "retrieved-evidence" else "not_applicable"
                                ),
                                "top_k": top_k if evidence_source == "retrieved-evidence" else "",
                                "model": model,
                                "evidence_source": evidence_source,
                                "answer": answer,
                                "source_label": f"[S{label}]",
                                "source_found": "no",
                                "claim_or_sentence": "",
                                "support_rating": "",
                                "relevance_rating": "",
                                "citation_notes": "",
                            }
                        )
                    completed += 1
                    print(f"  saved ({time.monotonic() - started:.1f}s)", flush=True)
                except Exception as exc:
                    writer.writerow(
                        _error_row(
                            question,
                            mode,
                            top_k,
                            model,
                            f"{type(exc).__name__}: {exc}",
                            evidence_source,
                            elapsed=time.monotonic() - started,
                        )
                    )
                    failed += 1
                    print(f"  error: {exc}", file=sys.stderr, flush=True)
                handle.flush()
                citation_handle.flush()

    print(f"Evidence source: {evidence_source}")
    print(f"Saved {completed} answers and {failed} errors to {output}")
    print(f"Saved cited-passage review rows to {citation_output}")


def _error_row(
    question: dict[str, object],
    mode: str,
    top_k: int,
    model: str,
    error: str,
    evidence_source: str,
    elapsed: float = 0.0,
) -> dict[str, object]:
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "document": question["document"],
        "answerability": question["answerability"],
        "evidence_source": evidence_source,
        "retrieval_mode": mode,
        "top_k": top_k if evidence_source == "retrieved-evidence" else "",
        "context_chunk_count": "",
        "model": model,
        "embedding_model": rag.EMBEDDING_MODEL,
        "chunk_target_words": rag.CHUNK_TARGET_WORDS,
        "chunk_max_words": rag.CHUNK_MAX_WORDS,
        "chunk_overlap_words": rag.CHUNK_OVERLAP_WORDS,
        "temperature": 0,
        "prompt_version": PROMPT_VERSION,
        "elapsed_seconds": f"{elapsed:.2f}",
        "error": error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and save answers for questions in a benchmark CSV."
    )
    parser.add_argument("benchmark", type=Path, help="CSV containing benchmark questions")
    parser.add_argument(
        "--evidence-source",
        choices=EVIDENCE_SOURCES,
        default="retrieved-evidence",
        help=(
            "use passages retrieved at run time or all relevant_chunk_ids from the "
            "benchmark (default: retrieved-evidence)"
        ),
    )
    parser.add_argument(
        "--retrieval-modes",
        nargs="+",
        choices=RETRIEVAL_MODES,
        default=["hybrid"],
        help="retrieval modes to evaluate (default: hybrid)",
    )
    parser.add_argument(
        "--top-k", type=int, default=6, help="number of chunks given to the model (default: 6)"
    )
    parser.add_argument("--model", default=rag.OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_generation_results/generation_outputs.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the output file if it exists"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        questions = load_questions(args.benchmark, args.evidence_source)
        evaluate(
            questions,
            list(dict.fromkeys(args.retrieval_modes)),
            args.top_k,
            args.model,
            args.output,
            args.overwrite,
            args.evidence_source,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
