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
        help="output Markdown path (defaults inside a dated run bundle)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", type=Path, help="dated results/YYYY-MM-DD bundle")
    source.add_argument("--results", type=Path, help="release result directory")
    parser.add_argument(
        "--hardware-results",
        type=Path,
        help="optional CUDA-event result directory used for the hardware table",
    )
    arguments = parser.parse_args()

    if arguments.run_dir:
        release_results = arguments.run_dir / "release"
        timing_results = arguments.run_dir / "cuda-timing"
        hardware_results = timing_results if timing_results.is_dir() else release_results
        report_path = arguments.report or arguments.run_dir / "report" / "performance_evaluation.md"
    else:
        release_results = arguments.results or rag_root / "results"
        hardware_results = arguments.hardware_results or release_results
        report_path = arguments.report or rag_root / "docs" / "performance_evaluation.md"

    if arguments.hardware_results:
        hardware_results = arguments.hardware_results

    application = read_rows(release_results / "rag_application_layer.csv")
    quality = read_rows(release_results / "rag_quality_gate.csv")
    stages = read_rows(release_results / "rag_stage_layer.csv")
    memory = read_rows(release_results / "rag_memory_layer.csv")
    hardware = read_rows(hardware_results / "rag_hardware_layer.csv")
    thermal_path = release_results / "rag_thermal_layer.csv"
    thermal = read_rows(thermal_path) if thermal_path.exists() else []
    manifest = json.loads(
        (release_results / "rag_run_manifest.json").read_text(encoding="utf-8")
    )
    template_path = rag_root / "docs" / "performance_evaluation.md"
    document = (
        report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else template_path.read_text(encoding="utf-8")
    )

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
    hardware_latency_rows = [{
        "Backend": backend_label(row["Platform"]),
        "Precision": row["Type"],
        "Batch": row["Batch"],
        "Trials": row["Trials"] or "—",
        "Samples": row["Samples"] or "—",
        "Batch P50 (µs)": row["Device batch P50 us"] or "—",
        "Batch P95 (µs)": row["Device batch P95 us"] or "—",
        "Batch P99 (µs)": row["Device batch P99 us"] or "—",
        "Device time (µs/query)": row["Device us/query"] or "—",
        "H2D P50 (µs)": row["H2D P50 us"] or "—",
        "H2D P95 (µs)": row.get("H2D P95 us", "") or "—",
        "H2D P99 (µs)": row.get("H2D P99 us", "") or "—",
        "Conversion P50 (µs)": row["Conversion P50 us"] or "—",
        "Conversion P95 (µs)": row.get("Conversion P95 us", "") or "—",
        "Conversion P99 (µs)": row.get("Conversion P99 us", "") or "—",
        "GEMM P50 (µs)": row["GEMM P50 us"] or "—",
        "GEMM P95 (µs)": row.get("GEMM P95 us", "") or "—",
        "GEMM P99 (µs)": row.get("GEMM P99 us", "") or "—",
        "Reduction P50 (µs)": row["Reduction P50 us"] or "—",
        "Reduction P95 (µs)": row.get("Reduction P95 us", "") or "—",
        "Reduction P99 (µs)": row.get("Reduction P99 us", "") or "—",
        "D2H P50 (µs)": row["D2H P50 us"] or "—",
        "D2H P95 (µs)": row.get("D2H P95 us", "") or "—",
        "D2H P99 (µs)": row.get("D2H P99 us", "") or "—",
        "Device/host (%)": row["Device/host %"] or "—",
    } for row in hardware]
    hardware_efficiency_rows = [{
        "Backend": backend_label(row["Platform"]),
        "Precision": row["Type"],
        "Batch": row["Batch"],
        "Transfer (B/query)": row["Transfer bytes/query"] or "—",
        "H2D (GB/s)": row["Effective H2D GB/s"] or "—",
        "D2H (GB/s)": row["Effective D2H GB/s"] or "—",
        "Effective GOP/s": row["Effective GOP/s"] or "—",
        "Effective GB/s": row["Effective GB/s"] or "—",
        "CPU normalization P50 (µs)": row["CPU normalization P50 us"] or "—",
        "Buffer preparation P50 (µs)": row["Buffer preparation P50 us"] or "—",
        "Result construction P50 (µs)": row["Host result construction P50 us"] or "—",
    } for row in hardware]
    thermal_rows = [{
        "Backend": backend_label(row["Platform"]),
        "Precision": row["Type"],
        "Batch": row["Batch"],
        "Provider": row["Provider"],
        "Power rail": row.get("Power rail", "") or "—",
        "Samples/errors": f"{row['Samples']}/{row['Sampling errors']}",
        "Initial temperature (°C)": row["Initial temp C"] or "—",
        "Mean temperature (°C)": row["Mean temp C"] or "—",
        "Maximum temperature (°C)": row["Max temp C"] or "—",
        "Thermal rise (°C)": row["Thermal rise C"] or "—",
        "Throttled samples": row["Throttled samples"] or "—",
        "Mean power (W)": row["Mean power W"] or "—",
        "Mean core clock (MHz)": row["Mean core clock MHz"] or "—",
        "Measured energy (J)": row.get("Measured energy J", "") or "—",
        "Max sample gap (s)": row.get("Max sample gap s", "") or "—",
    } for row in thermal]

    efficiency_heading = "## Hardware efficiency and host overhead"
    if efficiency_heading not in document:
        thermal_heading = "## Thermal telemetry"
        document = document.replace(
            thermal_heading,
            efficiency_heading
            + "\n\n"
            + "| Backend | Precision | Batch | Transfer (B/query) | H2D (GB/s) | D2H (GB/s) | Effective GOP/s | Effective GB/s | CPU normalization P50 (µs) | Buffer preparation P50 (µs) | Result construction P50 (µs) |\n"
            + "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n\n"
            + thermal_heading,
        )

    tables = [
        ("## Run configuration", ["Field", "Value"], run_configuration),
        ("## Quality gate", ["Backend", "Precision", "Queries", "Top-1 agreement (%)", "Gate agreement (%)", "Mean score error", "Max score error", "Status"], quality_rows),
        ("## Application performance", ["Backend", "Precision", "Batch", "Trials", "Requests", "Request", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Throughput mean (query/s)", "Throughput CV (%)", "Energy (J/query)"], application_rows),
        ("## Startup", ["Backend", "Precision", "Batch", "Bank load (ms)", "Backend init (ms)", "First request (ms)", "Cold start (ms)"], startup_rows),
        ("## RAG stages", ["Backend", "Precision", "Batch", "Path", "Stage", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Critical path (%)"], stage_rows),
        ("## Storage footprint", ["Backend", "Precision", "Entries", "Bank file", "Key data", "Offset table", "Mean payload"], storage_rows),
        ("## Runtime memory", ["Backend", "Precision", "Batch", "Peak host RSS", "Persistent device", "Peak device", "Workspace"], runtime_rows),
        ("## Hardware", ["Backend", "Precision", "Batch", "Trials", "Samples", "Batch P50 (µs)", "Batch P95 (µs)", "Batch P99 (µs)", "Device time (µs/query)", "H2D P50 (µs)", "H2D P95 (µs)", "H2D P99 (µs)", "Conversion P50 (µs)", "Conversion P95 (µs)", "Conversion P99 (µs)", "GEMM P50 (µs)", "GEMM P95 (µs)", "GEMM P99 (µs)", "Reduction P50 (µs)", "Reduction P95 (µs)", "Reduction P99 (µs)", "D2H P50 (µs)", "D2H P95 (µs)", "D2H P99 (µs)", "Device/host (%)"], hardware_latency_rows),
        (efficiency_heading, ["Backend", "Precision", "Batch", "Transfer (B/query)", "H2D (GB/s)", "D2H (GB/s)", "Effective GOP/s", "Effective GB/s", "CPU normalization P50 (µs)", "Buffer preparation P50 (µs)", "Result construction P50 (µs)"], hardware_efficiency_rows),
        ("## Thermal telemetry", ["Backend", "Precision", "Batch", "Provider", "Power rail", "Samples/errors", "Initial temperature (°C)", "Mean temperature (°C)", "Maximum temperature (°C)", "Thermal rise (°C)", "Throttled samples", "Mean power (W)", "Mean core clock (MHz)", "Measured energy (J)", "Max sample gap (s)"], thermal_rows),
    ]
    for heading, header, rows in tables:
        document = replace_table(document, heading, header, rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(document, encoding="utf-8", newline="\n")
    print(f"updated {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
