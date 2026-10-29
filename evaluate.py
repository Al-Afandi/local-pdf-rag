"""Run retrieval benchmarks and save every ranked chunk for inspection."""

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
ANSWERABILITY_LABELS = {"answerable", "unanswerable", "uncertain"}


def parse_relevant_ids(value: str, row_number: int) -> list[str]:
    """Parse multiple relevance labels separated by | or ;."""
    ids = list(dict.fromkeys(part.strip() for part in re.split(r"[;|]", value) if part.strip()))
    if not ids:
        raise ValueError(
            f"Benchmark row {row_number} has no relevant_chunk_ids. "
            "Add every chunk known to contain useful evidence, separated by |."
        )
    return ids


def load_benchmark(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"Benchmark CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question", "relevant_chunk_ids"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Benchmark CSV is missing required column(s): "
                + ", ".join(sorted(missing))
                + ". Required columns: question, relevant_chunk_ids; "
                "optional columns: question_id, document."
            )

        questions: list[dict[str, object]] = []
        for row_number, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()
            if not question:
                raise ValueError(f"Benchmark row {row_number} has an empty question.")
            question_id = (row.get("question_id") or "").strip() or f"q{row_number - 1:03d}"
            document = (row.get("document") or "").strip() or None
            raw_ids = (row.get("relevant_chunk_ids") or "").strip()
            raw_answerability = (row.get("answerability") or "").strip().casefold()
            if raw_answerability and raw_answerability not in ANSWERABILITY_LABELS:
                labels = ", ".join(sorted(ANSWERABILITY_LABELS))
                raise ValueError(
                    f"Benchmark row {row_number} has invalid answerability "
                    f"{raw_answerability!r}. Use one of: {labels}."
                )
            if raw_answerability:
                answerability = raw_answerability
            elif raw_ids:
                answerability = "answerable"
            else:
                raise ValueError(
                    f"Benchmark row {row_number} has no relevant_chunk_ids. "
                    "Mark it answerability=unanswerable if the source was checked "
                    "and contains no evidence, or uncertain if it still needs review."
                )
            relevant_ids = (
                parse_relevant_ids(raw_ids, row_number)
                if raw_ids
                else []
            )
            if answerability == "answerable" and not relevant_ids:
                raise ValueError(
                    f"Benchmark row {row_number} is marked answerable but has no "
                    "relevant_chunk_ids."
                )
            if answerability != "answerable" and relevant_ids:
                raise ValueError(
                    f"Benchmark row {row_number} is marked {answerability} but "
                    "also lists relevant_chunk_ids."
                )
            questions.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "document": document,
                    "answerability": answerability,
                    "relevant_chunk_ids": relevant_ids,
                }
            )

    if not questions:
        raise ValueError("Benchmark CSV has a header but no question rows.")
    return questions


def validate_relevance_labels(questions: list[dict[str, object]]) -> None:
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
        doc_name = question["document"]
        for chunk_id in question["relevant_chunk_ids"]:
            if chunk_id not in indexed:
                raise ValueError(
                    f"{question['question_id']}: relevant chunk {chunk_id!r} "
                    "is not in the current index. Check the ID or re-add the PDF "
                    "if its chunking settings changed."
                )
            if doc_name and indexed[chunk_id] != doc_name:
                raise ValueError(
                    f"{question['question_id']}: relevant chunk {chunk_id!r} "
                    f"belongs to {indexed[chunk_id]!r}, not the selected document {doc_name!r}."
                )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    questions: list[dict[str, object]],
    modes: list[str],
    retrieval_depth: int,
    metric_k: int,
    output_dir: Path,
) -> None:
    if retrieval_depth < 1:
        raise ValueError("--retrieval-depth must be at least 1.")
    if metric_k < 1 or metric_k > retrieval_depth:
        raise ValueError("--metric-k must be between 1 and --retrieval-depth.")

    output_dir.mkdir(parents=True, exist_ok=True)
    hit_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    per_mode: dict[str, list[dict[str, object]]] = defaultdict(list)

    for question in questions:
        question_id = str(question["question_id"])
        question_text = str(question["question"])
        document = question["document"]
        relevant_ids = set(question["relevant_chunk_ids"])
        answerability = str(question["answerability"])

        for mode in modes:
            hits = rag.retrieve_by_mode(
                question_text,
                top_k=retrieval_depth,
                document_name=document,
                retrieval_mode=mode,
            )
            is_scored = answerability == "answerable"
            found_ids = [hit.chunk_id for hit in hits if hit.chunk_id in relevant_ids]
            found_at_k = [hit for hit in hits[:metric_k] if hit.chunk_id in relevant_ids]
            missing_ids = sorted(relevant_ids - set(found_ids))
            first_rank = min((hit.rank for hit in found_at_k), default=None)
            recall = len({hit.chunk_id for hit in found_at_k}) / len(relevant_ids) if is_scored else None
            precision = (
                len(found_at_k) / min(metric_k, len(hits)) if hits else 0.0
            ) if is_scored else None
            reciprocal_rank = (1 / first_rank if first_rank is not None else 0.0) if is_scored else None

            for hit in hits:
                hit_rows.append(
                    {
                        "question_id": question_id,
                        "question": question_text,
                        "document_filter": document or "",
                        "answerability": answerability,
                        "retrieval_mode": mode,
                        "rank": hit.rank,
                        "score": f"{hit.score:.8f}",
                        "chunk_id": hit.chunk_id,
                        "is_relevant": hit.chunk_id in relevant_ids if is_scored else "",
                        "file_name": hit.file_name,
                        "page_number": hit.page_number,
                        "section": hit.section,
                        "text": hit.text,
                    }
                )

            metric = {
                "question_id": question_id,
                "question": question_text,
                "document_filter": document or "",
                "answerability": answerability,
                "retrieval_mode": mode,
                "retrieved_count": len(hits),
                "relevant_count": len(relevant_ids) if is_scored else "",
                f"relevant_found_at_{metric_k}": len(found_at_k) if is_scored else "",
                f"recall_at_{metric_k}": f"{recall:.6f}" if is_scored else "",
                f"precision_at_{metric_k}": f"{precision:.6f}" if is_scored else "",
                f"mrr_at_{metric_k}": f"{reciprocal_rank:.6f}" if is_scored else "",
                "first_relevant_rank": first_rank or "",
                f"all_relevant_found_at_{metric_k}": (
                    len(found_at_k) == len(relevant_ids) if is_scored else ""
                ),
                "relevant_chunk_ids": " | ".join(sorted(relevant_ids)),
                "found_relevant_chunk_ids_at_metric_k": " | ".join(
                    hit.chunk_id for hit in found_at_k
                ) if is_scored else "",
                "relevant_ids_not_retrieved": " | ".join(missing_ids) if is_scored else "",
            }
            metric_rows.append(metric)
            if is_scored:
                per_mode[mode].append(
                    {
                        "recall": recall,
                        "precision": precision,
                        "mrr": reciprocal_rank,
                        "success": bool(found_at_k),
                        "complete": len(found_at_k) == len(relevant_ids),
                    }
                )
                print(
                    f"{question_id} | {mode:14} | "
                    f"Recall@{metric_k}={recall:.2f} | "
                    f"first relevant rank={first_rank or 'not found'}"
                )
            else:
                print(f"{question_id} | {mode:14} | {answerability}; ranking metrics skipped")

    metric_k_fields = [
        "question_id", "question", "document_filter", "answerability", "retrieval_mode",
        "retrieved_count", "relevant_count", f"relevant_found_at_{metric_k}",
        f"recall_at_{metric_k}", f"precision_at_{metric_k}", f"mrr_at_{metric_k}",
        "first_relevant_rank", f"all_relevant_found_at_{metric_k}",
        "relevant_chunk_ids", "found_relevant_chunk_ids_at_metric_k",
        "relevant_ids_not_retrieved",
    ]
    write_csv(
        output_dir / "retrieved_chunks.csv",
        [
            "question_id", "question", "document_filter", "answerability", "retrieval_mode", "rank",
            "score", "chunk_id", "is_relevant", "file_name", "page_number",
            "section", "text",
        ],
        hit_rows,
    )
    write_csv(output_dir / "question_metrics.csv", metric_k_fields, metric_rows)

    summary_rows: list[dict[str, object]] = []
    for mode in modes:
        values = per_mode[mode]
        count = len(values)
        answerability_counts = {
            label: sum(question["answerability"] == label for question in questions)
            for label in ANSWERABILITY_LABELS
        }
        summary_rows.append(
            {
                "retrieval_mode": mode,
                "answerable_question_count": count,
                "unanswerable_question_count": answerability_counts["unanswerable"],
                "uncertain_question_count": answerability_counts["uncertain"],
                f"mean_recall_at_{metric_k}": f"{sum(v['recall'] for v in values) / count:.6f}" if count else "",
                f"mean_precision_at_{metric_k}": f"{sum(v['precision'] for v in values) / count:.6f}" if count else "",
                f"mrr_at_{metric_k}": f"{sum(v['mrr'] for v in values) / count:.6f}" if count else "",
                f"success_rate_at_{metric_k}": f"{sum(v['success'] for v in values) / count:.6f}" if count else "",
                f"complete_coverage_rate_at_{metric_k}": f"{sum(v['complete'] for v in values) / count:.6f}" if count else "",
            }
        )
    write_csv(
        output_dir / "mode_summary.csv",
        [
            "retrieval_mode", "answerable_question_count", "unanswerable_question_count",
            "uncertain_question_count", f"mean_recall_at_{metric_k}",
            f"mean_precision_at_{metric_k}", f"mrr_at_{metric_k}",
            f"success_rate_at_{metric_k}", f"complete_coverage_rate_at_{metric_k}",
        ],
        summary_rows,
    )

    print(f"\nSaved every retrieved chunk to: {output_dir / 'retrieved_chunks.csv'}")
    print(f"Saved per-question metrics to: {output_dir / 'question_metrics.csv'}")
    print(f"Saved mode averages to: {output_dir / 'mode_summary.csv'}")
    print(f"Metric cutoff: @{metric_k}; retrieval depth: @{retrieval_depth}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PDF retrieval against manually labeled relevant chunk IDs."
    )
    parser.add_argument("benchmark", type=Path, help="CSV with questions and relevant chunk IDs")
    parser.add_argument(
        "--retrieval-depth", type=int, default=10,
        help="number of ranked chunks to save per question (default: 10)",
    )
    parser.add_argument(
        "--metric-k", type=int, default=6,
        help="ranking cutoff for the reported metrics (default: 6)",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=RETRIEVAL_MODES, default=list(RETRIEVAL_MODES),
        help="retrieval modes to compare (default: all four; rerank modes may download a model)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation_results"),
        help="folder for CSV outputs (default: evaluation_results)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        questions = load_benchmark(args.benchmark)
        validate_relevance_labels(questions)
        evaluate(
            questions,
            args.modes,
            args.retrieval_depth,
            args.metric_k,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
