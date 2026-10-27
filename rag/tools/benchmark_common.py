"""Shared helpers for the RAG benchmark orchestration tools."""

from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path


RAW_FIELDS = [
    "backend",
    "batch",
    "mode",
    "iteration",
    "query_index",
    "query_count",
    "end_to_end_us",
    "vector_retrieval_us",
    "payload_retrieval_us",
    "cpu_normalization_us",
    "buffer_preparation_us",
    "host_result_construction_us",
    "h2d_us",
    "conversion_us",
    "gemm_us",
    "reduction_us",
    "d2h_us",
    "device_total_us",
    "h2d_bytes",
    "d2h_bytes",
    "device_timing_available",
    "cuda_timing_enabled",
    "nvtx_enabled",
    "payload_bytes",
    "peak_host_rss_bytes",
    "accepted",
    "id",
    "score",
]


def parse_backend(spec: str) -> tuple[str, str]:
    """Return (backend, precision) from NAME or NAME:PRECISION."""
    name, separator, precision = spec.partition(":")
    if not name:
        raise ValueError("backend name cannot be empty")
    precision = precision if separator else "fp32"
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"unsupported precision in backend specification: {spec}")
    return name, precision


def run_benchmark(
    executable: Path,
    bank: Path,
    queries: Path,
    expected: Path,
    backend_spec: str,
    *,
    batch: int = 1,
    warmup: int = 0,
    iterations: int = 1,
    threshold: float | None = None,
    raw_output: Path | None = None,
) -> list[dict[str, str]]:
    backend, precision = parse_backend(backend_spec)
    command = [
        str(executable),
        str(bank),
        str(queries),
        str(expected),
        "--backend",
        backend,
        "--precision",
        precision,
        "--batch",
        str(batch),
        "--warmup",
        str(warmup),
        "--iters",
        str(iterations),
    ]
    if threshold is not None:
        command.extend(["--threshold", str(threshold)])

    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if raw_output is not None:
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text(completed.stdout, encoding="utf-8", newline="\n")
    records = list(csv.DictReader(io.StringIO(completed.stdout)))
    if not records or list(records[0]) != RAW_FIELDS:
        raise RuntimeError(
            f"{backend} produced an unexpected rag_bench schema: "
            f"{list(records[0]) if records else 'no records'}"
        )
    return records


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
