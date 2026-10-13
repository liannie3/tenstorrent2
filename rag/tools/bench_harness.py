"""Run quality-gated RAG benchmarks and emit application/stage/memory results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
from datetime import datetime, timezone
from pathlib import Path

from benchmark_common import parse_backend, run_benchmark, write_csv
from quality_gate import FIELDS as QUALITY_FIELDS
from quality_gate import evaluate_backend


APPLICATION_FIELDS = [
    "Test", "Platform", "Mode", "Batch", "Type", "P50 ms", "P95 ms",
    "P99 ms", "Queries/s", "J/query", "Quality metric/value",
    "Top-1 agreement %", "Gate agreement %", "Cold-start ms",
    "First-request ms", "Bank-load ms", "Backend-init ms",
]

STAGE_FIELDS = [
    "Test", "Platform", "Batch", "Type", "Path", "Stage", "P50 ms",
    "P95 ms", "P99 ms", "Critical-path %",
]

MEMORY_FIELDS = [
    "Platform", "Batch", "Type", "Bank entries", "Bank-file bytes", "Raw-key bytes",
    "Offset-table bytes", "Mean returned payload bytes", "Peak host RSS",
    "Persistent device memory", "Peak device memory", "Workspace bytes",
]

HARDWARE_FIELDS = [
    "Test", "Platform", "Batch", "Type", "Device us/query", "Transfer us", "Transfer bytes/query",
    "Effective GOP/s", "Effective GB/s", "BW %", "Peak host RSS",
    "Persistent/peak device memory", "Workspace", "Bytes/vector", "Max entries",
]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def is_accepted(record: dict[str, str]) -> bool:
    return record["accepted"] in {"1", "true", "True"}


def measured_queries(records: list[dict[str, str]]) -> list[dict[str, str]]:
    return [record for record in records if record["mode"] == "measured"]


def submission_rows(
    records: list[dict[str, str]], path: str = "all"
) -> list[dict[str, str]]:
    selected = measured_queries(records)
    if path == "accepted":
        selected = [record for record in selected if is_accepted(record)]
    elif path == "rejected":
        selected = [record for record in selected if not is_accepted(record)]

    by_submission: dict[int, dict[str, str]] = {}
    for record in selected:
        by_submission.setdefault(int(record["iteration"]), record)
    return list(by_submission.values())


def phase_latency(records: list[dict[str, str]], mode: str) -> float:
    return next(
        float(row["end_to_end_us"]) for row in records if row["mode"] == mode
    )


def measured_throughput(records: list[dict[str, str]]) -> float:
    summary = next(row for row in records if row["mode"] == "measured_phase")
    elapsed_seconds = float(summary["end_to_end_us"]) / 1_000_000.0
    return int(summary["query_count"]) / elapsed_seconds


def latency_columns(rows: list[dict[str, str]], field: str) -> dict[str, str]:
    values = [float(row[field]) for row in rows]
    return {
        "P50 ms": f"{percentile(values, 0.50) / 1000.0:.6f}",
        "P95 ms": f"{percentile(values, 0.95) / 1000.0:.6f}",
        "P99 ms": f"{percentile(values, 0.99) / 1000.0:.6f}",
    }


def application_row(
    records: list[dict[str, str]],
    quality: dict,
    backend: str,
    precision: str,
    batch: int,
    path: str,
) -> dict[str, str | int]:
    rows = submission_rows(records, path)
    label = "Host-facing" if batch == 1 else "Batch request"
    row: dict[str, str | int] = {
        "Test": "Controlled" if precision == "bf16" else "Native",
        "Platform": backend,
        "Mode": f"{label}/{path}",
        "Batch": batch,
        "Type": precision.upper(),
        "Queries/s": f"{measured_throughput(records):.3f}" if path == "all" else "",
        "J/query": "",
        "Quality metric/value": "FP32 conformance",
        "Top-1 agreement %": quality["top1_agreement_pct"],
        "Gate agreement %": quality["gate_agreement_pct"],
        "Cold-start ms": f"{phase_latency(records, 'cold_start') / 1000.0:.6f}",
        "First-request ms": f"{phase_latency(records, 'first_request') / 1000.0:.6f}",
        "Bank-load ms": f"{phase_latency(records, 'bank_load') / 1000.0:.6f}",
        "Backend-init ms": f"{phase_latency(records, 'backend_init') / 1000.0:.6f}",
    }
    row.update(latency_columns(rows, "end_to_end_us"))
    return row


def stage_row(
    records: list[dict[str, str]],
    backend: str,
    precision: str,
    batch: int,
    path: str,
    stage: str,
    field: str,
) -> dict[str, str | int]:
    rows = submission_rows(records, path)
    stage_total = sum(float(row[field]) for row in rows)
    end_to_end_total = sum(float(row["end_to_end_us"]) for row in rows)
    row: dict[str, str | int] = {
        "Test": "Controlled" if precision == "bf16" else "Native",
        "Platform": backend,
        "Batch": batch,
        "Type": precision.upper(),
        "Path": path,
        "Stage": stage,
        "Critical-path %": f"{100.0 * stage_total / end_to_end_total:.3f}",
    }
    row.update(latency_columns(rows, field))
    return row


def bank_sizes(path: Path) -> dict[str, int]:
    with path.open("rb") as source:
        header = source.read(24)
    magic, count, text_dim, _, _ = struct.unpack("<8s4I", header)
    if magic != b"RAGBANK1":
        raise ValueError(f"invalid bank magic in {path}")
    return {
        "Bank entries": count,
        "Bank-file bytes": path.stat().st_size,
        "Raw-key bytes": count * text_dim * 4,
        "Offset-table bytes": count * 8,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-bench", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--backends", default="cpu")
    parser.add_argument("--batches", default="1")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    parser.add_argument(
        "--power-sampler",
        help="Reserved interface for a timestamp,watts sampler; energy remains blank",
    )
    arguments = parser.parse_args()
    if arguments.power_sampler:
        raise SystemExit(
            "--power-sampler is reserved for the hardware integration pass; "
            "energy measurement is not implemented"
        )

    backend_specs = [item.strip() for item in arguments.backends.split(",") if item.strip()]
    batches = [int(item) for item in arguments.batches.split(",") if item.strip()]
    if not backend_specs or not batches:
        raise SystemExit("at least one backend and batch size are required")
    bytes_per_query = 2048 * 4
    query_bytes = arguments.queries.stat().st_size
    if query_bytes == 0 or query_bytes % bytes_per_query:
        raise SystemExit("query corpus must contain complete 2048-float queries")
    query_count = query_bytes // bytes_per_query
    invalid_batches = [batch for batch in batches if batch <= 0 or query_count % batch]
    if invalid_batches:
        raise SystemExit(
            "query count must be divisible by every positive batch size; invalid: "
            + ", ".join(map(str, invalid_batches))
        )

    expected_document = json.loads(arguments.expected.read_text(encoding="utf-8"))
    threshold = float(expected_document["threshold"])

    quality_rows = [
        evaluate_backend(
            arguments.rag_bench, arguments.bank, arguments.queries,
            arguments.expected, spec,
        )
        for spec in backend_specs
    ]
    write_csv(arguments.results / "rag_quality_gate.csv", QUALITY_FIELDS, quality_rows)

    application_rows = []
    stage_rows = []
    memory_rows = []
    hardware_rows = []
    static_sizes = bank_sizes(arguments.bank)
    for backend_spec, quality in zip(backend_specs, quality_rows):
        if quality["passed"] != "true":
            print(f"skipping {backend_spec}: quality gate failed")
            continue
        backend, precision = parse_backend(backend_spec)
        for batch in batches:
            safe_backend = backend_spec.replace(":", "-")
            records = run_benchmark(
                arguments.rag_bench, arguments.bank, arguments.queries,
                arguments.expected, backend_spec,
                batch=batch, warmup=arguments.warmup, iterations=arguments.iters,
                threshold=threshold,
                raw_output=arguments.results / "raw" / f"{safe_backend}-b{batch}.csv",
            )
            application_rows.append(
                application_row(records, quality, backend, precision, batch, "all")
            )
            if batch == 1:
                for path in ("accepted", "rejected"):
                    if submission_rows(records, path):
                        application_rows.append(
                            application_row(records, quality, backend, precision, batch, path)
                        )

            vector_paths = ["all"]
            if batch == 1:
                vector_paths.extend(
                    path for path in ("accepted", "rejected")
                    if submission_rows(records, path)
                )
            for path in vector_paths:
                stage_rows.append(stage_row(
                    records, backend, precision, batch, path,
                    "Vector retrieval", "vector_retrieval_us",
                ))
            if batch == 1 and submission_rows(records, "accepted"):
                stage_rows.append(stage_row(
                    records, backend, precision, batch, "accepted",
                    "Payload retrieval", "payload_retrieval_us",
                ))

            payload_sizes = [
                int(row["payload_bytes"])
                for row in measured_queries(records) if is_accepted(row)
            ]
            measured_phase = next(
                row for row in records if row["mode"] == "measured_phase"
            )
            memory_rows.append({
                "Platform": backend,
                "Batch": batch,
                "Type": precision.upper(),
                **static_sizes,
                "Mean returned payload bytes": (
                    f"{sum(payload_sizes) / len(payload_sizes):.3f}"
                    if payload_sizes else "0"
                ),
                "Peak host RSS": measured_phase["peak_host_rss_bytes"],
                "Persistent device memory": "",
                "Peak device memory": "",
                "Workspace bytes": "",
            })
            hardware_row = {field: "" for field in HARDWARE_FIELDS}
            hardware_row.update({
                "Test": "Controlled" if precision == "bf16" else "Native",
                "Platform": backend,
                "Batch": batch,
                "Type": precision.upper(),
            })
            hardware_rows.append(hardware_row)

    write_csv(arguments.results / "rag_application_layer.csv", APPLICATION_FIELDS,
              application_rows)
    write_csv(arguments.results / "rag_stage_layer.csv", STAGE_FIELDS, stage_rows)
    write_csv(arguments.results / "rag_memory_layer.csv", MEMORY_FIELDS, memory_rows)
    write_csv(arguments.results / "rag_hardware_layer.csv", HARDWARE_FIELDS,
              hardware_rows)

    manifest = {
        "schema": "rag-benchmark-manifest-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "python": platform.python_version(),
        "backends": backend_specs,
        "batches": batches,
        "warmup_iterations": arguments.warmup,
        "measured_iterations": arguments.iters,
        "threshold": threshold,
        "files": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in {
                "executable": arguments.rag_bench,
                "bank": arguments.bank,
                "queries": arguments.queries,
                "expected": arguments.expected,
            }.items()
        },
    }
    manifest_path = arguments.results / "rag_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"wrote benchmark results to {arguments.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
