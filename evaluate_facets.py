"""Evaluate whether retrieved chunks cover the evidence facets of each question."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import rag


RETRIEVAL_MODES = ("dense", "hybrid", "rerank", "hybrid-rerank")
SOURCE_COVERAGE = {
    "answerable", "partially_answerable", "unanswerable", "uncertain"
}


def _split_ids(value: str) -> list[str]:
    return list(dict.fromkeys(
        item.strip() for item in re.split(r"[;|]", value) if item.strip()
    ))


def load_questions(path: Path) -> list[dict[str, str | None]]:
    if not path.is_file():
        raise ValueError(f"Benchmark CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question_id", "question", "answerability"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Benchmark CSV is missing columns: " + ", ".join(sorted(missing))
            )

        questions: list[dict[str, str | None]] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            question_id = (row.get("question_id") or "").strip()
            question = (row.get("question") or "").strip()
            answerability = (row.get("answerability") or "").strip().casefold()
            if not question_id or not question:
                raise ValueError(
                    f"Benchmark row {row_number} needs a question_id and question."
                )
            if question_id in seen_ids:
                raise ValueError(f"Duplicate question_id {question_id!r} in benchmark.")
            seen_ids.add(question_id)
            if answerability not in SOURCE_COVERAGE:
                choices = ", ".join(sorted(SOURCE_COVERAGE))
                raise ValueError(
                    f"{question_id}: answerability must be one of: {choices}."
                )
            questions.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "document": (row.get("document") or "").strip() or None,
                    "answerability": answerability,
                }
            )

    if not questions:
        raise ValueError("Benchmark CSV has no questions.")
    return questions


def load_facets(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.is_file():
        raise ValueError(f"Facet CSV not found: {path}")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "question_id", "facet_id", "facet_description", "supporting_chunk_ids"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Facet CSV is missing columns: " + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            question_id = (row.get("question_id") or "").strip()
            facet_id = (row.get("facet_id") or "").strip()
            description = (row.get("facet_description") or "").strip()
            chunk_ids = _split_ids(row.get("supporting_chunk_ids") or "")
            if not question_id or not facet_id or not description or not chunk_ids:
                raise ValueError(
                    f"Facet row {row_number} needs question_id, facet_id, "
                    "facet_description, and at least one supporting chunk ID."
                )
            key = (question_id, facet_id)
            if key in seen:
                raise ValueError(
                    f"Duplicate facet_id {facet_id!r} for question {question_id!r}."
                )
            seen.add(key)
            grouped[question_id].append(
                {
                    "facet_id": facet_id,
                    "description": description,
                    "chunk_ids": chunk_ids,
                }
            )
    return grouped


def validate_labels(
    questions: list[dict[str, str | None]],
    facets: dict[str, list[dict[str, object]]],
) -> None:
    question_ids = {question["question_id"] for question in questions}
    unknown_questions = set(facets) - question_ids
    if unknown_questions:
        raise ValueError(
            "Facet rows reference unknown question IDs: "
            + ", ".join(sorted(unknown_questions))
        )

    connection = rag._connect()
    try:
        rows = connection.execute(
            "SELECT chunks.chunk_id, documents.file_name "
            "FROM chunks JOIN documents USING (document_id)"
        ).fetchall()
    finally:
        connection.close()
    indexed = {row["chunk_id"]: row["file_name"] for row in rows}

    for question in questions:
        question_id = question["question_id"]
        answerability = question["answerability"]
        question_facets = facets.get(question_id, [])
        if answerability in {"answerable", "partially_answerable"} and not question_facets:
            raise ValueError(
                f"{question_id} is {answerability} but has no evidence facets."
            )
        if answerability in {"unanswerable", "uncertain"} and question_facets:
            raise ValueError(
                f"{question_id} is {answerability} and should not have facet labels."
            )

        for facet in question_facets:
            for chunk_id in facet["chunk_ids"]:
                if chunk_id not in indexed:
                    raise ValueError(
                        f"{question_id}/{facet['facet_id']}: chunk {chunk_id!r} "
                        "is not in the current index. Check its ID or re-add the PDF "
                        "if chunking settings changed."
                    )
                document = question["document"]
                if document and indexed[chunk_id] != document:
                    raise ValueError(
                        f"{question_id}/{facet['facet_id']}: chunk {chunk_id!r} "
                        f"belongs to {indexed[chunk_id]!r}, not {document!r}."
                    )


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    questions: list[dict[str, str | None]],
    facets_by_question: dict[str, list[dict[str, object]]],
    modes: list[str],
    retrieval_depth: int,
    metric_k: int,
    output_dir: Path,
) -> None:
    if retrieval_depth < 1:
        raise ValueError("--retrieval-depth must be at least 1.")
    if not 1 <= metric_k <= retrieval_depth:
        raise ValueError("--metric-k must be between 1 and --retrieval-depth.")

    output_dir.mkdir(parents=True, exist_ok=True)
    hit_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    mode_metrics: dict[str, list[dict[str, float | bool]]] = defaultdict(list)

    for question in questions:
        question_id = str(question["question_id"])
        question_text = str(question["question"])
        document = question["document"]
        answerability = str(question["answerability"])
        question_facets = facets_by_question.get(question_id, [])
        is_scored = answerability in {"answerable", "partially_answerable"}
        chunk_to_facets: dict[str, list[str]] = defaultdict(list)
        for facet in question_facets:
            for chunk_id in facet["chunk_ids"]:
                chunk_to_facets[chunk_id].append(str(facet["facet_id"]))

        for mode in modes:
            hits = rag.retrieve_by_mode(
                question_text,
                top_k=retrieval_depth,
                document_name=document,
                retrieval_mode=mode,
            )
            retrieved_at_k = {hit.chunk_id for hit in hits[:metric_k]}
            found_facets = [
                facet for facet in question_facets
                if retrieved_at_k.intersection(facet["chunk_ids"])
            ]
            facet_count = len(question_facets)
            facets_found = len(found_facets)
            facet_recall = facets_found / facet_count if is_scored else None
            first_rank = min(
                (hit.rank for hit in hits[:metric_k]
                 if hit.chunk_id in chunk_to_facets),
                default=None,
            )
            mrr = (1 / first_rank if first_rank else 0.0) if is_scored else None
            if not is_scored:
                coverage_status = "source_unanswerable" if answerability == "unanswerable" else "source_uncertain"
            elif facets_found == facet_count:
                coverage_status = "all_available_facets_found"
            elif facets_found:
                coverage_status = "some_available_facets_found"
            else:
                coverage_status = "no_available_facets_found"

            for hit in hits:
                matched_facets = chunk_to_facets.get(hit.chunk_id, [])
                hit_rows.append(
                    {
                        "question_id": question_id,
                        "question": question_text,
                        "document_filter": document or "",
                        "source_answerability": answerability,
                        "retrieval_mode": mode,
                        "rank": hit.rank,
                        "score": f"{hit.score:.8f}",
                        "chunk_id": hit.chunk_id,
                        "matched_facet_ids": " | ".join(matched_facets) if is_scored else "",
                        "supports_any_facet": bool(matched_facets) if is_scored else "",
                        "file_name": hit.file_name,
                        "page_number": hit.page_number,
                        "section": hit.section,
                        "text": hit.text,
                    }
                )

            metric_rows.append(
                {
                    "question_id": question_id,
                    "question": question_text,
                    "document_filter": document or "",
                    "source_answerability": answerability,
                    "retrieval_mode": mode,
                    "facet_count": facet_count if is_scored else "",
                    f"facets_found_at_{metric_k}": facets_found if is_scored else "",
                    f"facet_recall_at_{metric_k}": f"{facet_recall:.6f}" if is_scored else "",
                    "first_supporting_rank": first_rank or "",
                    f"mrr_at_{metric_k}": f"{mrr:.6f}" if is_scored else "",
                    f"evidence_coverage_at_{metric_k}": coverage_status,
                    "facets_found": " | ".join(str(f["facet_id"]) for f in found_facets) if is_scored else "",
                    "facets_not_found": " | ".join(
                        str(facet["facet_id"]) for facet in question_facets
                        if facet not in found_facets
                    ) if is_scored else "",
                }
            )

            if is_scored:
                mode_metrics[mode].append(
                    {
                        "facet_recall": facet_recall,
                        "mrr": mrr,
                        "any_facet": bool(facets_found),
                        "all_facets": facets_found == facet_count,
                    }
                )
                print(
                    f"{question_id} | {mode:14} | "
                    f"facet coverage@{metric_k}={facets_found}/{facet_count} | "
                    f"{coverage_status}"
                )
            else:
                print(f"{question_id} | {mode:14} | {coverage_status}; ranking metrics skipped")

    _write_csv(
        output_dir / "retrieved_facets.csv",
        [
            "question_id", "question", "document_filter", "source_answerability",
            "retrieval_mode", "rank", "score", "chunk_id", "matched_facet_ids",
            "supports_any_facet", "file_name", "page_number", "section", "text",
        ],
        hit_rows,
    )
    _write_csv(
        output_dir / "facet_question_metrics.csv",
        [
            "question_id", "question", "document_filter", "source_answerability",
            "retrieval_mode", "facet_count", f"facets_found_at_{metric_k}",
            f"facet_recall_at_{metric_k}", "first_supporting_rank", f"mrr_at_{metric_k}",
            f"evidence_coverage_at_{metric_k}", "facets_found", "facets_not_found",
        ],
        metric_rows,
    )

    source_counts = {
        label: sum(str(question["answerability"]) == label for question in questions)
        for label in SOURCE_COVERAGE
    }
    summary_rows: list[dict[str, object]] = []
    for mode in modes:
        values = mode_metrics[mode]
        count = len(values)
        summary_rows.append(
            {
                "retrieval_mode": mode,
                "fully_answerable_questions": source_counts["answerable"],
                "partially_answerable_questions": source_counts["partially_answerable"],
                "unanswerable_questions": source_counts["unanswerable"],
                "uncertain_questions": source_counts["uncertain"],
                "scored_question_count": count,
                f"mean_facet_recall_at_{metric_k}": f"{sum(v['facet_recall'] for v in values) / count:.6f}" if count else "",
                f"mrr_at_{metric_k}": f"{sum(v['mrr'] for v in values) / count:.6f}" if count else "",
                f"any_facet_found_rate_at_{metric_k}": f"{sum(v['any_facet'] for v in values) / count:.6f}" if count else "",
                f"all_facets_found_rate_at_{metric_k}": f"{sum(v['all_facets'] for v in values) / count:.6f}" if count else "",
            }
        )
    _write_csv(
        output_dir / "facet_mode_summary.csv",
        [
            "retrieval_mode", "fully_answerable_questions", "partially_answerable_questions",
            "unanswerable_questions", "uncertain_questions", "scored_question_count",
            f"mean_facet_recall_at_{metric_k}", f"mrr_at_{metric_k}",
            f"any_facet_found_rate_at_{metric_k}", f"all_facets_found_rate_at_{metric_k}",
        ],
        summary_rows,
    )
    print(f"\nSaved facet-level results in: {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval coverage of answer evidence facets."
    )
    parser.add_argument("benchmark", type=Path, help="CSV containing question IDs and source answerability")
    parser.add_argument("facets", type=Path, help="CSV mapping each question facet to supporting chunk IDs")
    parser.add_argument("--retrieval-depth", type=int, default=10)
    parser.add_argument("--metric-k", type=int, default=6)
    parser.add_argument(
        "--modes", nargs="+", choices=RETRIEVAL_MODES,
        default=list(RETRIEVAL_MODES),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation_facet_results")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        questions = load_questions(args.benchmark)
        facets = load_facets(args.facets)
        validate_labels(questions, facets)
        evaluate(
            questions, facets, args.modes, args.retrieval_depth,
            args.metric_k, args.output_dir,
        )
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
