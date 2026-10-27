"""Run quality-gated RAG benchmarks and emit application/stage/memory results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import struct
from datetime import datetime, timezone
from pathlib import Path

from benchmark_common import parse_backend, run_benchmark, write_csv
from quality_gate import FIELDS as QUALITY_FIELDS
from quality_gate import evaluate_backend


APPLICATION_FIELDS = [
    "Test", "Platform", "Mode", "Batch", "Type", "Trials", "Requests",
    "P50 ms", "P95 ms", "P99 ms", "Queries/s", "Throughput CV %", "J/query", "Quality metric/value",
    "Top-1 agreement %", "Gate agreement %", "Cold-start ms",
    "Cold-start SD ms", "First-request ms", "First-request SD ms",
    "Bank-load ms", "Bank-load SD ms", "Backend-init ms", "Backend-init SD ms",
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
    "Test", "Platform", "Batch", "Type", "Trials", "Samples",
    "Device batch P50 us", "Device batch P95 us", "Device batch P99 us",
    "Device us/query", "H2D P50 us", "Conversion P50 us", "GEMM P50 us",
    "Reduction P50 us", "D2H P50 us", "Transfer us", "Transfer bytes/query",
    "Effective H2D GB/s", "Effective D2H GB/s", "Effective GOP/s",
    "Effective GB/s", "BW %", "Device/host %", "CPU normalization P50 us",
    "Buffer preparation P50 us", "Host result construction P50 us", "Peak host RSS",
    "Persistent/peak device memory", "Workspace", "Bytes/vector", "Max entries",
]

DEFAULT_BATCHES = "1,2,4,8,16,32,64,128"


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

    by_submission: dict[tuple[str, int], dict[str, str]] = {}
    for record in selected:
        key = (record.get("_trial", "0"), int(record["iteration"]))
        by_submission.setdefault(key, record)
    return list(by_submission.values())


def phase_statistics(records: list[dict[str, str]], mode: str) -> tuple[float, float]:
    values = [
        float(row["end_to_end_us"]) / 1000.0
        for row in records if row["mode"] == mode
    ]
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def measured_throughputs(records: list[dict[str, str]]) -> list[float]:
    return [
        int(row["query_count"]) / (float(row["end_to_end_us"]) / 1_000_000.0)
        for row in records if row["mode"] == "measured_phase"
    ]


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
    throughputs = measured_throughputs(records)
    mean_throughput = statistics.fmean(throughputs)
    throughput_cv = (
        100.0 * statistics.stdev(throughputs) / mean_throughput
        if len(throughputs) > 1 else 0.0
    )
    phases = {
        mode: phase_statistics(records, mode)
        for mode in ("cold_start", "first_request", "bank_load", "backend_init")
    }
    label = "Host-facing" if batch == 1 else "Batch request"
    row: dict[str, str | int] = {
        "Test": "Controlled" if precision == "bf16" else "Native",
        "Platform": backend,
        "Mode": f"{label}/{path}",
        "Batch": batch,
        "Type": precision.upper(),
        "Trials": len(throughputs),
        "Requests": len(rows),
        "Queries/s": f"{mean_throughput:.3f}" if path == "all" else "",
        "Throughput CV %": f"{throughput_cv:.3f}" if path == "all" else "",
        "J/query": "",
        "Quality metric/value": "FP32 conformance",
        "Top-1 agreement %": quality["top1_agreement_pct"],
        "Gate agreement %": quality["gate_agreement_pct"],
        "Cold-start ms": f"{phases['cold_start'][0]:.6f}",
        "Cold-start SD ms": f"{phases['cold_start'][1]:.6f}",
        "First-request ms": f"{phases['first_request'][0]:.6f}",
        "First-request SD ms": f"{phases['first_request'][1]:.6f}",
        "Bank-load ms": f"{phases['bank_load'][0]:.6f}",
        "Bank-load SD ms": f"{phases['bank_load'][1]:.6f}",
        "Backend-init ms": f"{phases['backend_init'][0]:.6f}",
        "Backend-init SD ms": f"{phases['backend_init'][1]:.6f}",
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


def hardware_row(
    records: list[dict[str, str]],
    backend: str,
    precision: str,
    batch: int,
    bank_entries: int,
) -> dict[str, str | int]:
    row: dict[str, str | int] = {field: "" for field in HARDWARE_FIELDS}
    row.update({
        "Test": "Controlled" if precision == "bf16" else "Native",
        "Platform": backend,
        "Batch": batch,
        "Type": precision.upper(),
    })
    samples = [
        sample for sample in submission_rows(records)
        if sample.get("device_timing_available") in {"1", "true", "True"}
    ]
    if not samples:
        return row

    def values(field: str) -> list[float]:
        return [float(sample[field]) for sample in samples]

    def p50(field: str) -> float:
        return percentile(values(field), 0.50)

    device_values = values("device_total_us")
    trials = {sample.get("_trial", "0") for sample in samples}
    h2d = p50("h2d_us")
    d2h = p50("d2h_us")
    gemm = p50("gemm_us")
    transfer_bytes_per_query = statistics.fmean(
        (float(sample["h2d_bytes"]) + float(sample["d2h_bytes"])) / batch
        for sample in samples
    )
    similarity_operations = 2 * bank_entries * 2048 * batch
    key_bytes = bank_entries * 2048 * 4 * batch
    vector_p50 = p50("vector_retrieval_us")
    h2d_bytes = statistics.fmean(float(sample["h2d_bytes"]) for sample in samples)
    d2h_bytes = statistics.fmean(float(sample["d2h_bytes"]) for sample in samples)
    device_p50 = percentile(device_values, 0.50)
    row.update({
        "Trials": len(trials),
        "Samples": len(samples),
        "Device batch P50 us": f"{percentile(device_values, 0.50):.6f}",
        "Device batch P95 us": f"{percentile(device_values, 0.95):.6f}",
        "Device batch P99 us": f"{percentile(device_values, 0.99):.6f}",
        "Device us/query": f"{device_p50 / batch:.6f}",
        "H2D P50 us": f"{h2d:.6f}",
        "Conversion P50 us": f"{p50('conversion_us'):.6f}",
        "GEMM P50 us": f"{gemm:.6f}",
        "Reduction P50 us": f"{p50('reduction_us'):.6f}",
        "D2H P50 us": f"{d2h:.6f}",
        "Transfer us": f"{h2d + d2h:.6f}",
        "Transfer bytes/query": f"{transfer_bytes_per_query:.3f}",
        "Effective H2D GB/s": f"{h2d_bytes / (h2d * 1000.0):.6f}" if h2d else "",
        "Effective D2H GB/s": f"{d2h_bytes / (d2h * 1000.0):.6f}" if d2h else "",
        "Effective GOP/s": f"{similarity_operations / (gemm * 1000.0):.6f}" if gemm else "",
        "Effective GB/s": f"{key_bytes / (gemm * 1000.0):.6f}" if gemm else "",
        "Device/host %": f"{100.0 * device_p50 / vector_p50:.3f}" if vector_p50 else "",
        "CPU normalization P50 us": f"{p50('cpu_normalization_us'):.6f}",
        "Buffer preparation P50 us": f"{p50('buffer_preparation_us'):.6f}",
        "Host result construction P50 us": f"{p50('host_result_construction_us'):.6f}",
    })
    return row


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
    parser.add_argument("--batches", default=DEFAULT_BATCHES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
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
    if not backend_specs or not batches or arguments.trials <= 0:
        raise SystemExit("at least one backend, batch size, and trial are required")
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
    cuda_timing_enabled = False
    nvtx_enabled = False
    static_sizes = bank_sizes(arguments.bank)
    for backend_spec, quality in zip(backend_specs, quality_rows):
        if quality["passed"] != "true":
            print(f"skipping {backend_spec}: quality gate failed")
            continue
        backend, precision = parse_backend(backend_spec)
        for batch in batches:
            safe_backend = backend_spec.replace(":", "-")
            records = []
            for trial in range(1, arguments.trials + 1):
                print(
                    f"running {backend_spec} batch={batch} "
                    f"trial={trial}/{arguments.trials}",
                    flush=True,
                )
                trial_records = run_benchmark(
                    arguments.rag_bench, arguments.bank, arguments.queries,
                    arguments.expected, backend_spec,
                    batch=batch, warmup=arguments.warmup, iterations=arguments.iters,
                    threshold=threshold,
                    raw_output=(arguments.results / "raw" /
                                f"{safe_backend}-b{batch}-run{trial:02}.csv"),
                )
                for record in trial_records:
                    record["_trial"] = str(trial)
                records.extend(trial_records)
            cuda_timing_enabled = cuda_timing_enabled or any(
                row.get("cuda_timing_enabled") in {"1", "true", "True"}
                for row in records
            )
            nvtx_enabled = nvtx_enabled or any(
                row.get("nvtx_enabled") in {"1", "true", "True"}
                for row in records
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
            measured_phases = [
                row for row in records if row["mode"] == "measured_phase"
            ]
            memory_rows.append({
                "Platform": backend,
                "Batch": batch,
                "Type": precision.upper(),
                **static_sizes,
                "Mean returned payload bytes": (
                    f"{sum(payload_sizes) / len(payload_sizes):.3f}"
                    if payload_sizes else "0"
                ),
                "Peak host RSS": max(
                    int(row["peak_host_rss_bytes"]) for row in measured_phases
                ),
                "Persistent device memory": "",
                "Peak device memory": "",
                "Workspace bytes": "",
            })
            hardware_rows.append(hardware_row(
                records, backend, precision, batch, static_sizes["Bank entries"]
            ))

    write_csv(arguments.results / "rag_application_layer.csv", APPLICATION_FIELDS,
              application_rows)
    write_csv(arguments.results / "rag_stage_layer.csv", STAGE_FIELDS, stage_rows)
    write_csv(arguments.results / "rag_memory_layer.csv", MEMORY_FIELDS, memory_rows)
    write_csv(arguments.results / "rag_hardware_layer.csv", HARDWARE_FIELDS,
              hardware_rows)

    manifest = {
        "schema": "rag-benchmark-manifest-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "python": platform.python_version(),
        "backends": backend_specs,
        "batches": batches,
        "warmup_iterations": arguments.warmup,
        "measured_iterations": arguments.iters,
        "independent_trials": arguments.trials,
        "threshold": threshold,
        "instrumentation": {
            "cuda_event_timing": cuda_timing_enabled,
            "nvtx": nvtx_enabled,
        },
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
