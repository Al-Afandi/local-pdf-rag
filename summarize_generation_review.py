"""Summarize human answer and citation ratings from generation review CSVs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


ANSWER_METRICS = {
    "scope_fit": {"in_scope"},
    "factual_correctness": {"correct"},
    "answer_completeness": {"complete"},
    "citation_support_overall": {"full"},
    "citation_completeness": {"complete"},
}
UNANSWERABLE = {"unanswerable", "uncertain"}
ANSWERABLE = {"answerable", "partially_answerable", "partially answerable"}
EMPTY_LABELS = {"", "-", "n/a", "na", "not_applicable", "not applicable"}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Review CSV not found: {path}")
    if path.suffix.casefold() == ".csv":
        import csv

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.casefold() == ".xlsx":
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise ValueError("Reading .xlsx reviews requires openpyxl; install requirements.txt.") from exc
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            sheet = workbook.active
            rows = sheet.iter_rows(values_only=True)
            headers = [str(value).strip() if value is not None else "" for value in next(rows, ())]
            return [
                {
                    header: "" if value is None else str(value)
                    for header, value in zip(headers, values)
                    if header
                }
                for values in rows
                if any(value is not None for value in values)
            ]
        finally:
            workbook.close()
    raise ValueError(f"Unsupported review file format: {path.suffix}; use CSV or XLSX.")


def norm(value: str | None) -> str:
    return (value or "").strip().casefold()


def distribution(rows: list[dict[str, str]], field: str) -> Counter[str]:
    return Counter(
        label
        for row in rows
        if (label := norm(row.get(field))) not in EMPTY_LABELS
    )


def positive_rate(
    rows: list[dict[str, str]], field: str, accepted: set[str]
) -> tuple[int, int]:
    counts = distribution(rows, field)
    return sum(count for label, count in counts.items() if label in accepted), sum(counts.values())


def rate_text(numerator: int, denominator: int) -> str:
    if not denominator:
        return "not rated (0/0)"
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def fmt_distribution(counts: Counter[str]) -> str:
    if not counts:
        return "no labels"
    return ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))


def summarize_answers(path: Path) -> list[str]:
    rows = read_rows(path)
    answerable = [row for row in rows if norm(row.get("answerability")) in ANSWERABLE]
    unanswerable = [row for row in rows if norm(row.get("answerability")) in UNANSWERABLE]
    lines = [f"## {path.name}", "", f"Rows: {len(rows)}; answerable/partial: {len(answerable)}; unanswerable/uncertain: {len(unanswerable)}", ""]

    for field, accepted in ANSWER_METRICS.items():
        numerator, denominator = positive_rate(answerable, field, accepted)
        lines.append(f"- **{field}:** {rate_text(numerator, denominator)}; labels: {fmt_distribution(distribution(answerable, field))}")

    fully_rated = [row for row in answerable if all(norm(row.get(field)) not in EMPTY_LABELS for field in ANSWER_METRICS)]
    all_positive = sum(
        all(norm(row.get(field)) in accepted for field, accepted in ANSWER_METRICS.items())
        for row in fully_rated
    )
    lines.append(f"- **All five answer dimensions positive:** {rate_text(all_positive, len(fully_rated))}")

    abstention_counts = distribution(unanswerable, "abstention_correctness")
    correct_abstentions = sum(
        count for label, count in abstention_counts.items()
        if label in {"correct", "correct_abstention", "correct abstention"}
    )
    lines.append(
        f"- **Correct abstention:** {rate_text(correct_abstentions, sum(abstention_counts.values()))}; "
        f"labels: {fmt_distribution(abstention_counts)}"
    )
    return lines


def summarize_citations(path: Path) -> list[str]:
    if not path.is_file():
        return [f"Citation review: companion file not found ({path.name})."]
    rows = read_rows(path)
    lines = [f"### Citations: {path.name}", "", f"Rows: {len(rows)}", ""]
    for field, title in (
        ("source_found", "Source found"),
        ("support_rating", "Claim support"),
        ("relevance_rating", "Citation relevance"),
    ):
        counts = distribution(rows, field)
        lines.append(f"- **{title}:** {fmt_distribution(counts)}")
    return lines


def build_report(answer_files: list[Path]) -> str:
    sections: list[str] = ["# Human generation review summary", ""]
    for answer_path in answer_files:
        sections.extend(summarize_answers(answer_path))
        citation_path = answer_path.with_name(f"{answer_path.stem}_citations{answer_path.suffix}")
        sections.extend([""] + summarize_citations(citation_path) + [""])
    sections.extend(
        [
            "## How to read this summary",
            "",
            "Answer-level rates use only nonblank labels in the answerable or partially answerable rows. The combined answer rate counts a row only when all five answer dimensions are labeled and positive. Correct abstention is reported separately for unanswerable and uncertain rows. Citation support, source-found, and relevance distributions are separate; a tangential citation is not automatically an unsupported claim.",
            "",
        ]
    )
    return "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare manually reviewed generation answer and citation CSVs."
    )
    parser.add_argument("answers", nargs="+", type=Path, help="one or more answer review CSVs")
    parser.add_argument("--output", type=Path, help="optional Markdown report path")
    args = parser.parse_args()
    try:
        report = build_report(args.answers)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
            print(f"Saved review summary to {args.output}")
        else:
            print(report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
