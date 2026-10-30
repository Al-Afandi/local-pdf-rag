"""Evaluate retrieval coverage using alternate evidence options per facet."""

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
OPTION_LEVELS = {"complete", "partial"}


def _parse_ids(value: str, separator: str) -> list[str]:
    return list(dict.fromkeys(
        item.strip() for item in value.split(separator) if item.strip()
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
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    seen_options: set[tuple[str, str, str]] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "question_id", "facet_id", "facet_description", "evidence_option_id",
            "option_level", "required_chunk_ids",
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
            option_id = (row.get("evidence_option_id") or "").strip()
            level = (row.get("option_level") or "").strip().casefold()
            chunk_ids = _parse_ids(row.get("required_chunk_ids") or "", "&")
            if not all((question_id, facet_id, description, option_id, chunk_ids)):
                raise ValueError(
                    f"Facet row {row_number} needs question_id, facet_id, "
                    "description, evidence_option_id, and required chunk IDs."
                )
            if level not in OPTION_LEVELS:
                raise ValueError(
                    f"Facet row {row_number}: option_level must be complete or partial."
                )
            key = (question_id, facet_id, option_id)
            if key in seen_options:
                raise ValueError(
                    f"Duplicate evidence option {option_id!r} for "
                    f"{question_id}/{facet_id}."
                )
            seen_options.add(key)

            facets = grouped[question_id]
            if facet_id not in facets:
                facets[facet_id] = {
                    "facet_id": facet_id,
                    "description": description,
                    "options": [],
                }
            elif facets[facet_id]["description"] != description:
                raise ValueError(
                    f"Facet {question_id}/{facet_id} has inconsistent descriptions."
                )
            facets[facet_id]["options"].append(
                {
                    "option_id": option_id,
                    "level": level,
                    "chunk_ids": chunk_ids,
                }
            )

    return {qid: list(facets.values()) for qid, facets in grouped.items()}


def validate_labels(
    questions: list[dict[str, str | None]],
    facets_by_question: dict[str, list[dict[str, object]]],
) -> None:
    question_ids = {question["question_id"] for question in questions}
    unknown_questions = set(facets_by_question) - question_ids
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
        facets = facets_by_question.get(question_id, [])
        if answerability in {"answerable", "partially_answerable"} and not facets:
            raise ValueError(f"{question_id} is {answerability} but has no evidence facets.")
        if answerability in {"unanswerable", "uncertain"} and facets:
            raise ValueError(f"{question_id} is {answerability} and should have no facet rows.")

        for facet in facets:
            for option in facet["options"]:
                for chunk_id in option["chunk_ids"]:
                    if chunk_id not in indexed:
                        raise ValueError(
                            f"{question_id}/{facet['facet_id']}/{option['option_id']}: "
                            f"chunk {chunk_id!r} is not in the current index. Check the "
                            "ID or re-add the PDF if chunking settings changed."
                        )
                    document = question["document"]
                    if document and indexed[chunk_id] != document:
                        raise ValueError(
                            f"{question_id}/{facet['facet_id']}: chunk {chunk_id!r} "
                            f"belongs to {indexed[chunk_id]!r}, not {document!r}."
                        )


def assess_facet(
    facet: dict[str, object], ranks: dict[str, int]
) -> dict[str, object]:
    complete_options: list[tuple[int, str]] = []
    partial_options: list[tuple[int, str]] = []
    touched_options: list[str] = []

    for option in facet["options"]:
        chunk_ids = set(option["chunk_ids"])
        present = chunk_ids.intersection(ranks)
        if not present:
            continue
        touched_options.append(str(option["option_id"]))
        if chunk_ids.issubset(ranks):
            ready_rank = max(ranks[chunk_id] for chunk_id in chunk_ids)
            target = complete_options if option["level"] == "complete" else partial_options
            target.append((ready_rank, str(option["option_id"])))
        else:
            partial_options.append(
                (min(ranks[chunk_id] for chunk_id in present), str(option["option_id"]))
            )

    if complete_options:
        level = "complete"
        ready_rank, option_id = min(complete_options)
    elif partial_options:
        level = "partial"
        ready_rank, option_id = min(partial_options)
    else:
        level = "none"
        ready_rank, option_id = None, ""

    return {
        "level": level,
        "ready_rank": ready_rank,
        "option_id": option_id,
        "touched_options": touched_options,
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
    retrieved_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    mode_values: dict[str, list[dict[str, float | bool]]] = defaultdict(list)

    for question in questions:
        question_id = str(question["question_id"])
        question_text = str(question["question"])
        document = question["document"]
        source_status = str(question["answerability"])
        facets = facets_by_question.get(question_id, [])
        scored = source_status in {"answerable", "partially_answerable"}

        for mode in modes:
            hits = rag.retrieve_by_mode(
                question_text, top_k=retrieval_depth, document_name=document,
                retrieval_mode=mode,
            )
            all_ranks = {hit.chunk_id: hit.rank for hit in hits}
            cutoff_ranks = {
                hit.chunk_id: hit.rank for hit in hits if hit.rank <= metric_k
            }
            facet_results = [assess_facet(facet, cutoff_ranks) for facet in facets]
            complete_count = sum(result["level"] == "complete" for result in facet_results)
            partial_count = sum(result["level"] == "partial" for result in facet_results)
            evidence_count = complete_count + partial_count
            facet_count = len(facets)

            if not scored:
                status = "source_unanswerable" if source_status == "unanswerable" else "source_uncertain"
            elif facet_count and complete_count == facet_count:
                status = "all_facets_complete"
            elif evidence_count:
                status = "partial_evidence"
            else:
                status = "no_evidence_found"

            complete_ranks: list[int] = []
            for facet in facets:
                option_ranks = []
                for option in facet["options"]:
                    ids = set(option["chunk_ids"])
                    if option["level"] == "complete" and ids.issubset(all_ranks):
                        option_ranks.append(max(all_ranks[chunk_id] for chunk_id in ids))
                if option_ranks:
                    complete_ranks.append(min(option_ranks))
                else:
                    complete_ranks = []
                    break
            sufficient_rank = max(complete_ranks) if complete_ranks else None
            mrr_sufficient = (
                1 / sufficient_rank
                if sufficient_rank is not None and sufficient_rank <= metric_k
                else 0.0
            ) if scored else None

            for hit in hits:
                matched: list[str] = []
                options_here: list[str] = []
                for facet in facets:
                    for option in facet["options"]:
                        if hit.chunk_id in option["chunk_ids"]:
                            matched.append(str(facet["facet_id"]))
                            options_here.append(str(option["option_id"]))
                retrieved_rows.append(
                    {
                        "question_id": question_id,
                        "question": question_text,
                        "document_filter": document or "",
                        "source_answerability": source_status,
                        "retrieval_mode": mode,
                        "rank": hit.rank,
                        "score": f"{hit.score:.8f}",
                        "chunk_id": hit.chunk_id,
                        "matched_facet_ids": " | ".join(dict.fromkeys(matched)) if scored else "",
                        "evidence_option_ids": " | ".join(dict.fromkeys(options_here)) if scored else "",
                        "file_name": hit.file_name,
                        "page_number": hit.page_number,
                        "section": hit.section,
                        "text": hit.text,
                    }
                )

            evidence_coverage = evidence_count / facet_count if scored else ""
            complete_facet_rate = complete_count / facet_count if scored else ""
            metric_rows.append(
                {
                    "question_id": question_id,
                    "question": question_text,
                    "document_filter": document or "",
                    "source_answerability": source_status,
                    "retrieval_mode": mode,
                    "facet_count": facet_count if scored else "",
                    f"facets_with_evidence_at_{metric_k}": evidence_count if scored else "",
                    f"complete_facets_at_{metric_k}": complete_count if scored else "",
                    f"partial_facets_at_{metric_k}": partial_count if scored else "",
                    f"facet_evidence_coverage_at_{metric_k}": f"{evidence_coverage:.6f}" if scored else "",
                    f"complete_facet_rate_at_{metric_k}": f"{complete_facet_rate:.6f}" if scored else "",
                    "first_sufficient_rank": sufficient_rank or "",
                    f"mrr_sufficient_at_{metric_k}": f"{mrr_sufficient:.6f}" if scored else "",
                    f"evidence_sufficiency_at_{metric_k}": status,
                    "facet_results": " | ".join(
                        f"{facet['facet_id']}:{result['level']}"
                        for facet, result in zip(facets, facet_results)
                    ) if scored else "",
                }
            )

            if scored:
                mode_values[mode].append(
                    {
                        "evidence_coverage": float(evidence_coverage),
                        "complete_facet_rate": float(complete_facet_rate),
                        "mrr_sufficient": float(mrr_sufficient),
                        "sufficient": status == "all_facets_complete",
                        "partial": status == "partial_evidence",
                        "none": status == "no_evidence_found",
                    }
                )
                print(f"{question_id} | {mode:14} | {status} | "
                      f"complete facets {complete_count}/{facet_count}; "
                      f"partial facets {partial_count}/{facet_count}")
            else:
                print(f"{question_id} | {mode:14} | {status}; ranking metrics skipped")

    write_csv(
        output_dir / "retrieved_facets.csv",
        ["question_id", "question", "document_filter", "source_answerability",
         "retrieval_mode", "rank", "score", "chunk_id", "matched_facet_ids",
         "evidence_option_ids", "file_name", "page_number", "section", "text"],
        retrieved_rows,
    )
    write_csv(
        output_dir / "facet_question_metrics.csv",
        ["question_id", "question", "document_filter", "source_answerability",
         "retrieval_mode", "facet_count", f"facets_with_evidence_at_{metric_k}",
         f"complete_facets_at_{metric_k}", f"partial_facets_at_{metric_k}",
         f"facet_evidence_coverage_at_{metric_k}", f"complete_facet_rate_at_{metric_k}",
         "first_sufficient_rank", f"mrr_sufficient_at_{metric_k}",
         f"evidence_sufficiency_at_{metric_k}", "facet_results"],
        metric_rows,
    )

    source_counts = {
        label: sum(question["answerability"] == label for question in questions)
        for label in SOURCE_COVERAGE
    }
    summary_rows: list[dict[str, object]] = []
    for mode in modes:
        values = mode_values[mode]
        count = len(values)
        summary_rows.append(
            {
                "retrieval_mode": mode,
                "fully_answerable_questions": source_counts["answerable"],
                "partially_answerable_questions": source_counts["partially_answerable"],
                "unanswerable_questions": source_counts["unanswerable"],
                "uncertain_questions": source_counts["uncertain"],
                "scored_question_count": count,
                f"mean_facet_evidence_coverage_at_{metric_k}": f"{sum(v['evidence_coverage'] for v in values) / count:.6f}" if count else "",
                f"mean_complete_facet_rate_at_{metric_k}": f"{sum(v['complete_facet_rate'] for v in values) / count:.6f}" if count else "",
                f"mrr_sufficient_at_{metric_k}": f"{sum(v['mrr_sufficient'] for v in values) / count:.6f}" if count else "",
                f"sufficient_answer_rate_at_{metric_k}": f"{sum(v['sufficient'] for v in values) / count:.6f}" if count else "",
                f"partial_evidence_rate_at_{metric_k}": f"{sum(v['partial'] for v in values) / count:.6f}" if count else "",
                f"no_evidence_rate_at_{metric_k}": f"{sum(v['none'] for v in values) / count:.6f}" if count else "",
            }
        )
    write_csv(
        output_dir / "facet_mode_summary.csv",
        ["retrieval_mode", "fully_answerable_questions", "partially_answerable_questions",
         "unanswerable_questions", "uncertain_questions", "scored_question_count",
         f"mean_facet_evidence_coverage_at_{metric_k}",
         f"mean_complete_facet_rate_at_{metric_k}", f"mrr_sufficient_at_{metric_k}",
         f"sufficient_answer_rate_at_{metric_k}", f"partial_evidence_rate_at_{metric_k}",
         f"no_evidence_rate_at_{metric_k}"],
        summary_rows,
    )
    print(f"\nSaved evidence-option results in: {output_dir.resolve()}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval coverage of complete and partial evidence options."
    )
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("facets", type=Path)
    parser.add_argument("--retrieval-depth", type=int, default=10)
    parser.add_argument("--metric-k", type=int, default=6)
    parser.add_argument(
        "--modes", nargs="+", choices=RETRIEVAL_MODES, default=list(RETRIEVAL_MODES)
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluation_option_results")
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
