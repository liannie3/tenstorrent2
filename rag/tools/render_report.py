"""Render benchmark CSV rows into the RAG performance evaluation report."""

from __future__ import annotations

import argparse
import csv
import json
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
    heading_index = document.index(heading)
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


def backend_label(value: str) -> str:
    return {
        "cpu": "CPU reference",
        "cuda-reference": "CUDA reference",
        "cuda-optimized": "CUDA optimized",
    }.get(value, value)


def request_label(value: str) -> str:
    kind, path = value.split("/", maxsplit=1)
    kind = "Single request" if kind == "Host-facing" else kind
    return f"{kind} · {path}"


def mean_sd(row: dict[str, str], prefix: str) -> str:
    return f"{row[prefix + ' ms']} ± {row[prefix + ' SD ms']}"


def byte_size(value: str) -> str:
    size = int(value)
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MiB"
    if size >= 1024:
        return f"{size / 1024:.2f} KiB"
    return f"{size} B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    rag_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--report", type=Path,
        default=rag_root / "docs" / "performance_evaluation.md",
    )
    parser.add_argument("--results", type=Path, default=rag_root / "results")
    arguments = parser.parse_args()

    application = read_rows(arguments.results / "rag_application_layer.csv")
    quality = read_rows(arguments.results / "rag_quality_gate.csv")
    stages = read_rows(arguments.results / "rag_stage_layer.csv")
    memory = read_rows(arguments.results / "rag_memory_layer.csv")
    hardware = read_rows(arguments.results / "rag_hardware_layer.csv")
    thermal_path = arguments.results / "rag_thermal_layer.csv"
    thermal = read_rows(thermal_path) if thermal_path.exists() else []
    manifest = json.loads(
        (arguments.results / "rag_run_manifest.json").read_text(encoding="utf-8")
    )
    document = arguments.report.read_text(encoding="utf-8")

    query_count = quality[0]["query_count"] if quality else ""
    run_configuration = [
        {"Field": "Generated (UTC)", "Value": manifest["created_utc"]},
        {"Field": "Host", "Value": manifest["host"]},
        {"Field": "Python", "Value": manifest["python"]},
        {"Field": "Backends", "Value": ", ".join(manifest["backends"])},
        {"Field": "Batch sizes", "Value": ", ".join(map(str, manifest["batches"]))},
        {"Field": "Query corpus", "Value": f"{query_count} queries"},
        {"Field": "Memory bank", "Value": f"{memory[0]['Bank entries']} entries"},
        {"Field": "Similarity threshold", "Value": f"{manifest['threshold']:.3f}"},
        {"Field": "Independent trials", "Value": str(manifest["independent_trials"])},
        {"Field": "Warmup iterations", "Value": str(manifest["warmup_iterations"])},
        {
            "Field": "Measured corpus passes/trial",
            "Value": str(manifest["measured_iterations"]),
        },
    ]
    quality_rows = [{
        "Backend": backend_label(row["backend"]),
        "Precision": row["precision"].upper(),
        "Queries": row["query_count"],
        "Top-1 agreement (%)": row["top1_agreement_pct"],
        "Gate agreement (%)": row["gate_agreement_pct"],
        "Mean score error": row["mean_abs_score_error"],
        "Max score error": row["max_abs_score_error"],
        "Status": "PASS" if row["passed"] == "true" else "FAIL",
    } for row in quality]
    application_rows = [{
        "Backend": backend_label(row["Platform"]),
        "Precision": row["Type"],
        "Batch": row["Batch"],
        "Trials": row["Trials"],
        "Requests": row["Requests"],
        "Request": request_label(row["Mode"]),
        "P50 (ms)": row["P50 ms"],
        "P95 (ms)": row["P95 ms"],
        "P99 (ms)": row["P99 ms"],
        "Throughput mean (query/s)": row["Queries/s"] or "—",
        "Throughput CV (%)": row["Throughput CV %"] or "—",
        "Energy (J/query)": row["J/query"] or "—",
    } for row in application]
    all_application = [row for row in application if row["Mode"].endswith("/all")]
    startup_rows = [{
        "Backend": backend_label(row["Platform"]),
        "Precision": row["Type"],
        "Batch": row["Batch"],
        "Bank load (ms)": mean_sd(row, "Bank-load"),
        "Backend init (ms)": mean_sd(row, "Backend-init"),
        "First request (ms)": mean_sd(row, "First-request"),
        "Cold start (ms)": mean_sd(row, "Cold-start"),
    } for row in all_application]
    stage_rows = [{
        "Backend": backend_label(row["Platform"]),
        "Precision": row["Type"],
        "Batch": row["Batch"],
        "Path": row["Path"],
        "Stage": row["Stage"],
        "P50 (ms)": row["P50 ms"],
        "P95 (ms)": row["P95 ms"],
        "P99 (ms)": row["P99 ms"],
        "Critical path (%)": row["Critical-path %"],
    } for row in stages]
    storage_rows = []
    seen_storage = set()
    for row in memory:
        key = (row["Platform"], row["Type"])
        if key in seen_storage:
            continue
        seen_storage.add(key)
        storage_rows.append({
            "Backend": backend_label(row["Platform"]),
            "Precision": row["Type"],
            "Entries": row["Bank entries"],
            "Bank file": byte_size(row["Bank-file bytes"]),
            "Key data": byte_size(row["Raw-key bytes"]),
            "Offset table": byte_size(row["Offset-table bytes"]),
            "Mean payload": byte_size(str(round(float(row["Mean returned payload bytes"])))),
        })
    runtime_rows = [{
        "Backend": backend_label(row["Platform"]),
        "Precision": row["Type"],
        "Batch": row["Batch"],
        "Peak host RSS": byte_size(row["Peak host RSS"]),
        "Persistent device": row["Persistent device memory"] or "—",
        "Peak device": row["Peak device memory"] or "—",
        "Workspace": row["Workspace bytes"] or "—",
    } for row in memory]
    hardware_rows = [{
        "Backend": backend_label(row["Platform"]),
        "Precision": row["Type"],
        "Batch": row["Batch"],
        "Device time (µs/query)": row["Device us/query"] or "—",
        "Transfer time (µs)": row["Transfer us"] or "—",
        "Transfer (B/query)": row["Transfer bytes/query"] or "—",
        "Effective GOP/s": row["Effective GOP/s"] or "—",
        "Effective GB/s": row["Effective GB/s"] or "—",
        "Bandwidth (%)": row["BW %"] or "—",
    } for row in hardware]
    thermal_rows = [{
        "Backend": backend_label(row["Platform"]),
        "Precision": row["Type"],
        "Batch": row["Batch"],
        "Provider": row["Provider"],
        "Samples/errors": f"{row['Samples']}/{row['Sampling errors']}",
        "Initial temperature (°C)": row["Initial temp C"] or "—",
        "Mean temperature (°C)": row["Mean temp C"] or "—",
        "Maximum temperature (°C)": row["Max temp C"] or "—",
        "Thermal rise (°C)": row["Thermal rise C"] or "—",
        "Throttled samples": row["Throttled samples"] or "—",
        "Mean power (W)": row["Mean power W"] or "—",
        "Mean core clock (MHz)": row["Mean core clock MHz"] or "—",
    } for row in thermal]

    tables = [
        ("## Run configuration", ["Field", "Value"], run_configuration),
        ("## Quality gate", ["Backend", "Precision", "Queries", "Top-1 agreement (%)", "Gate agreement (%)", "Mean score error", "Max score error", "Status"], quality_rows),
        ("## Application performance", ["Backend", "Precision", "Batch", "Trials", "Requests", "Request", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Throughput mean (query/s)", "Throughput CV (%)", "Energy (J/query)"], application_rows),
        ("## Startup", ["Backend", "Precision", "Batch", "Bank load (ms)", "Backend init (ms)", "First request (ms)", "Cold start (ms)"], startup_rows),
        ("## RAG stages", ["Backend", "Precision", "Batch", "Path", "Stage", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Critical path (%)"], stage_rows),
        ("## Storage footprint", ["Backend", "Precision", "Entries", "Bank file", "Key data", "Offset table", "Mean payload"], storage_rows),
        ("## Runtime memory", ["Backend", "Precision", "Batch", "Peak host RSS", "Persistent device", "Peak device", "Workspace"], runtime_rows),
        ("## Hardware", ["Backend", "Precision", "Batch", "Device time (µs/query)", "Transfer time (µs)", "Transfer (B/query)", "Effective GOP/s", "Effective GB/s", "Bandwidth (%)"], hardware_rows),
        ("## Thermal telemetry", ["Backend", "Precision", "Batch", "Provider", "Samples/errors", "Initial temperature (°C)", "Mean temperature (°C)", "Maximum temperature (°C)", "Thermal rise (°C)", "Throttled samples", "Mean power (W)", "Mean core clock (MHz)"], thermal_rows),
    ]
    for heading, header, rows in tables:
        document = replace_table(document, heading, header, rows)

    arguments.report.write_text(document, encoding="utf-8", newline="\n")
    print(f"updated {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
