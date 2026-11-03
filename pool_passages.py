"""Export a deduplicated pool of passages for benchmark annotation.

For each question, chunks retrieved by multiple modes are written once and
sorted by their chunk ordinal, approximating the paper's reading order.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import rag


RETRIEVAL_MODES = ("dense", "hybrid", "rerank", "hybrid-rerank")
OUTPUT_FIELDS = (
    "question_id",
    "document",
    "question_type",
    "source_answerability",
    "facet_plan",
    "question",
    "chunk_id",
    "passage",
    "title",
    "page_number",
)


def load_questions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Question CSV not found: {path}")

    questions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError("Question CSV must include a 'question' column.")

        for row_number, row in enumerate(reader, start=2):
            question_id = (row.get("question_id") or str(row_number - 1)).strip()
            question = (row.get("question") or "").strip()
            document_filter = (
                row.get("document") or row.get("document_filter") or ""
            ).strip()
            if not question:
                raise ValueError(f"Question row {row_number} has an empty question.")
            if question_id in seen_ids:
                raise ValueError(
                    f"Duplicate question_id {question_id!r} in question CSV."
                )
            seen_ids.add(question_id)
            questions.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "document_filter": document_filter,
                    "question_type": (row.get("question_type") or "").strip(),
                    "source_answerability": (
                        row.get("source_answerability")
                        or row.get("answerability")
                        or ""
                    ).strip(),
                    "facet_plan": (row.get("facet_plan") or "").strip(),
                }
            )

    if not questions:
        raise ValueError("Question CSV has no questions.")
    return questions


def chunk_order(chunk_id: str) -> tuple[int, str]:
    """Sort by the numeric cNNNN suffix, with a stable fallback."""
    suffix = chunk_id.rsplit("-c", 1)[-1]
    try:
        return int(suffix), chunk_id
    except ValueError:
        return 2**31 - 1, chunk_id


def collect_unique_passages(
    questions: list[dict[str, str]],
    modes: list[str],
    top_k: int,
) -> list[dict[str, str | int]]:
    output_rows: list[dict[str, str | int]] = []
    total_searches = len(questions) * len(modes)
    completed_searches = 0

    for question in questions:
        # Deduplicate within a question/document across all selected modes.
        unique: dict[tuple[str, str], dict[str, str | int]] = {}

        for mode in modes:
            hits = rag.retrieve_by_mode(
                question["question"],
                top_k=top_k,
                document_name=question["document_filter"] or None,
                retrieval_mode=mode,
            )
            completed_searches += 1
            print(
                f"[{completed_searches}/{total_searches}] "
                f"{question['question_id']} {mode}: {len(hits)} passage(s)"
            )

            for hit in hits:
                key = (hit.file_name, hit.chunk_id)
                if key not in unique:
                    unique[key] = {
                        "document": hit.file_name,
                        "chunk_id": hit.chunk_id,
                        "passage": hit.text,
                        "title": hit.section or "",
                        "page_number": hit.page_number,
                    }

        ordered = sorted(
            unique.values(),
            key=lambda row: (
                str(row["document"]).casefold(),
                chunk_order(str(row["chunk_id"])),
            ),
        )
        for passage in ordered:
            output_rows.append(
                {
                    "question_id": question["question_id"],
                    "document": passage["document"],
                    "question_type": question["question_type"],
                    "source_answerability": question["source_answerability"],
                    "facet_plan": question["facet_plan"],
                    "question": question["question"],
                    "chunk_id": passage["chunk_id"],
                    "passage": passage["passage"],
                    "title": passage["title"],
                    "page_number": passage["page_number"],
                }
            )
        print(
            f"  {len(ordered)} unique passage(s), sorted by chunk ID "
            f"for paper order"
        )

    return output_rows


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve passages for a CSV question list, deduplicate chunks "
            "across selected modes, and export passages in document order."
        )
    )
    parser.add_argument(
        "questions",
        type=Path,
        help="input CSV with question and optional document/annotation columns",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="passages retrieved per question and mode (default: 10)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=RETRIEVAL_MODES,
        default=list(RETRIEVAL_MODES),
        help="retrieval modes to pool (default: all four)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/passage_pool.csv"),
        help="output CSV path (default: benchmarks/passage_pool.csv)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k < 1:
        print("Error: --top-k must be at least 1.", file=sys.stderr)
        return 2

    try:
        questions = load_questions(args.questions)
        rows = collect_unique_passages(questions, args.modes, args.top_k)
        write_csv(args.output, rows)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {len(rows)} unique passage rows for {len(questions)} questions "
        f"to {args.output}."
    )
    print(
        "Output columns follow the requested order; passages are sorted by "
        "chunk ID within each question."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
