"""Render benchmark CSV rows into the RAG performance evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def markdown_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def replace_table(
    document: str,
    heading: str,
    header: list[str],
    rows: list[dict[str, str]],
) -> str:
    # Result-table headings intentionally repeat the evaluation-layer headings;
    # update the final occurrence under "RAG result tables".
    heading_index = document.rindex(heading)
    table_start = document.index("| " + header[0] + " |", heading_index)
    table_end = table_start
    while table_end < len(document) and document[table_end] == "|":
        newline = document.find("\n", table_end)
        table_end = len(document) if newline == -1 else newline + 1

    rendered = [
        markdown_row(header),
        markdown_row(["---"] * len(header)),
        *(markdown_row([row.get(column, "") for column in header]) for row in rows),
    ]
    return document[:table_start] + "\n".join(rendered) + "\n" + document[table_end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    rag_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--report",
        type=Path,
        default=rag_root / "docs" / "performance_evaluation.md",
    )
    parser.add_argument(
        "--results", type=Path, default=rag_root / "results"
    )
    arguments = parser.parse_args()

    application = read_rows(arguments.results / "rag_application_layer.csv")
    stages = read_rows(arguments.results / "rag_stage_layer.csv")
    memory = read_rows(arguments.results / "rag_memory_layer.csv")
    hardware = read_rows(arguments.results / "rag_hardware_layer.csv")
    document = arguments.report.read_text(encoding="utf-8")
    application_header = [
        "Test", "Platform", "Mode", "Batch", "Type", "P50 ms", "P95 ms",
        "P99 ms", "Queries/s", "J/query", "Quality metric/value",
        "Top-1 agreement %", "Gate agreement %",
    ]
    stage_header = [
        "Test", "Platform", "Batch", "Type", "Path", "Stage", "P50 ms",
        "P95 ms", "P99 ms", "Critical-path %",
    ]
    memory_header = [
        "Platform", "Batch", "Type", "Bank entries", "Bank-file bytes", "Raw-key bytes",
        "Offset-table bytes", "Mean returned payload bytes", "Peak host RSS",
        "Persistent device memory", "Peak device memory", "Workspace bytes",
    ]
    hardware_header = [
        "Test", "Platform", "Batch", "Type", "Device us/query", "Transfer us", "Transfer bytes/query",
        "Effective GOP/s", "Effective GB/s", "BW %", "Peak host RSS",
        "Persistent/peak device memory", "Workspace", "Bytes/vector", "Max entries",
    ]
    document = replace_table(
        document, "### Application", application_header, application
    )
    document = replace_table(
        document, "### RAG stages", stage_header, stages
    )
    document = replace_table(
        document, "### RAG memory", memory_header, memory
    )
    document = replace_table(
        document, "### Hardware", hardware_header, hardware
    )
    arguments.report.write_text(document, encoding="utf-8", newline="\n")
    print(f"updated {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
