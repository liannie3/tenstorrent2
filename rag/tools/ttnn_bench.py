#!/usr/bin/env python3
"""rag_bench-compatible command adapter for the TT-NN reference backend.

The process opens the device once, keeps the key bank resident, and reuses the
searcher for cold, warmup, and measured requests. TT-NN v0.80 currently returns
the score matrix for deterministic host-side reduction; this is deliberately a
correctness reference rather than the optimized Metalium design.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Callable, Sequence, TextIO

from benchmark_common import RAW_FIELDS
from ttnn_backend import Match, SearchResult, TtnnReferenceSearcher

TEXT_DIMENSION = 2048
VISUAL_FLOATS = 101 * 1024
HEADER = struct.Struct("<8s4I")
OFFSET = struct.Struct("<Q")
DESCRIPTION_SIZE = struct.Struct("<I")


def elapsed_us(start_ns: int, finish_ns: int) -> float:
    return (finish_ns - start_ns) / 1_000.0


def load_queries(path: Path) -> list[float]:
    raw = path.read_bytes()
    bytes_per_query = TEXT_DIMENSION * 4
    if not raw or len(raw) % bytes_per_query:
        raise ValueError("query corpus must contain complete 2048-float queries")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


class PayloadReader:
    """Minimal payload reader matching rag::Bank's retrieval timing boundary."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as source:
            raw_header = source.read(HEADER.size)
            if len(raw_header) != HEADER.size:
                raise ValueError("truncated RAGBANK1 header")
            magic, entries, dimension, tokens, visual_dimension = HEADER.unpack(raw_header)
            if magic != b"RAGBANK1" or dimension != TEXT_DIMENSION:
                raise ValueError("unsupported bank format")
            if (tokens, visual_dimension) != (101, 1024) or entries <= 0:
                raise ValueError("unsupported bank dimensions")
            source.seek(entries * dimension * 4, os.SEEK_CUR)
            raw_offsets = source.read(entries * OFFSET.size)
            if len(raw_offsets) != entries * OFFSET.size:
                raise ValueError("truncated bank offset table")
        self.entries = entries
        self.offsets = struct.unpack(f"<{entries}Q", raw_offsets)

    def load_size(self, entry_id: int) -> int:
        if entry_id < 0 or entry_id >= self.entries:
            raise ValueError("match ID is outside the bank")
        with self.path.open("rb") as source:
            source.seek(self.offsets[entry_id])
            raw_size = source.read(DESCRIPTION_SIZE.size)
            if len(raw_size) != DESCRIPTION_SIZE.size:
                raise ValueError("truncated payload description length")
            description_size = DESCRIPTION_SIZE.unpack(raw_size)[0]
            expected = description_size + VISUAL_FLOATS * 4
            if len(source.read(expected)) != expected:
                raise ValueError("truncated bank payload")
        return expected


def empty_record(backend: str, mode: str) -> dict[str, object]:
    record: dict[str, object] = {field: "" for field in RAW_FIELDS}
    record.update({
        "backend": backend,
        "batch": 1,
        "mode": mode,
        "iteration": 0,
        "query_index": -1,
        "query_count": 0,
        "end_to_end_us": 0.0,
        "vector_retrieval_us": 0.0,
        "payload_retrieval_us": 0.0,
        "device_memory_available": 0,
        "device_timing_available": 0,
        "cuda_timing_enabled": 0,
        "nvtx_enabled": 0,
        "payload_bytes": 0,
        "peak_host_rss_bytes": 0,
        "accepted": 0,
        "id": 0,
        "score": -1.0,
    })
    return record


def apply_match(record: dict[str, object], match: Match) -> None:
    record.update({
        "accepted": int(match.accepted),
        "id": match.id,
        "score": f"{match.score:.9g}",
    })


def apply_timings(record: dict[str, object], result: SearchResult) -> None:
    timing = result.timings
    for field in (
        "cpu_normalization_us", "buffer_preparation_us", "device_total_us",
        "h2d_bytes", "d2h_bytes", "persistent_device_bytes",
        "peak_device_bytes", "workspace_bytes",
    ):
        value = getattr(timing, field, None)
        if value is not None:
            record[field] = value
    record["device_memory_available"] = int(all(
        getattr(timing, field, None) is not None for field in (
            "persistent_device_bytes", "peak_device_bytes", "workspace_bytes"
        )
    ))
    # device_total_us is a synchronized host observation, not a Tensix timer.
    record["device_timing_available"] = 0


def execute_submission(
    searcher: TtnnReferenceSearcher,
    payloads: PayloadReader,
    queries: Sequence[float],
    threshold: float,
) -> tuple[SearchResult, list[int], float, float, float]:
    query_count = len(queries) // TEXT_DIMENSION
    start = time.perf_counter_ns()
    result = searcher.search_batch(queries, query_count, threshold)
    if len(result.matches) != query_count:
        raise RuntimeError("TT-NN backend returned the wrong result count")
    vector_finish = time.perf_counter_ns()
    payload_sizes = [
        payloads.load_size(match.id) if match.accepted else 0
        for match in result.matches
    ]
    finish = time.perf_counter_ns()
    has_payload = any(match.accepted for match in result.matches)
    return (
        result,
        payload_sizes,
        elapsed_us(start, finish),
        elapsed_us(start, vector_finish),
        elapsed_us(vector_finish, finish) if has_payload else 0.0,
    )


def peak_host_rss_bytes() -> int:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return 0


def run(
    arguments: argparse.Namespace,
    output: TextIO,
    searcher_factory: Callable[[Path, int], TtnnReferenceSearcher] = TtnnReferenceSearcher,
) -> None:
    process_start = time.perf_counter_ns()
    bank_start = time.perf_counter_ns()
    payloads = PayloadReader(arguments.bank)
    bank_finish = time.perf_counter_ns()
    query_start = time.perf_counter_ns()
    queries = load_queries(arguments.queries)
    query_finish = time.perf_counter_ns()
    query_count = len(queries) // TEXT_DIMENSION
    if query_count % arguments.batch:
        raise ValueError("query count must be divisible by batch size")

    backend_start = time.perf_counter_ns()
    if arguments.packed_bank is None and not arguments.tracy_signposts:
        searcher = searcher_factory(arguments.bank, arguments.device)
    else:
        searcher = searcher_factory(
            arguments.bank, arguments.device,
            packed_bank_path=arguments.packed_bank,
            max_batch=max(1, arguments.batch),
            tracy_signposts=arguments.tracy_signposts,
        )
    backend_finish = time.perf_counter_ns()
    writer = csv.DictWriter(output, fieldnames=RAW_FIELDS, lineterminator="\n")
    writer.writeheader()

    def phase(mode: str, start: int, finish: int) -> None:
        record = empty_record("ttnn-reference", mode)
        record["end_to_end_us"] = elapsed_us(start, finish)
        writer.writerow(record)

    phase("bank_load", bank_start, bank_finish)
    phase("query_load", query_start, query_finish)
    phase("backend_init", backend_start, backend_finish)

    try:
        first = queries[:TEXT_DIMENSION]
        first_result, first_sizes, end_us, vector_us, payload_us = execute_submission(
            searcher, payloads, first, arguments.threshold
        )
        first_finish = time.perf_counter_ns()
        cold = empty_record("ttnn-reference", "cold_start")
        cold.update({
            "query_count": 1,
            "end_to_end_us": elapsed_us(process_start, first_finish),
            "payload_bytes": first_sizes[0],
        })
        apply_match(cold, first_result.matches[0])
        apply_timings(cold, first_result)
        writer.writerow(cold)
        first_record = empty_record("ttnn-reference", "first_request")
        first_record.update({
            "query_count": 1,
            "end_to_end_us": end_us,
            "vector_retrieval_us": vector_us,
            "payload_retrieval_us": payload_us,
            "payload_bytes": first_sizes[0],
        })
        apply_match(first_record, first_result.matches[0])
        apply_timings(first_record, first_result)
        writer.writerow(first_record)

        def corpus_pass(record: bool, submission: int) -> int:
            for offset in range(0, query_count, arguments.batch):
                begin = offset * TEXT_DIMENSION
                end = (offset + arguments.batch) * TEXT_DIMENSION
                result, sizes, total_us, search_us, payload_us = execute_submission(
                    searcher, payloads, queries[begin:end], arguments.threshold
                )
                if record:
                    for index, (match, payload_size) in enumerate(
                        zip(result.matches, sizes)
                    ):
                        row = empty_record("ttnn-reference", "measured")
                        row.update({
                            "batch": arguments.batch,
                            "iteration": submission,
                            "query_index": offset + index,
                            "query_count": arguments.batch,
                            "end_to_end_us": total_us,
                            "vector_retrieval_us": search_us,
                            "payload_retrieval_us": payload_us,
                            "payload_bytes": payload_size,
                        })
                        apply_match(row, match)
                        apply_timings(row, result)
                        writer.writerow(row)
                    submission += 1
            return submission

        for _ in range(arguments.warmup):
            corpus_pass(False, 0)
        measured_start = time.perf_counter_ns()
        submissions = 0
        for _ in range(arguments.iters):
            submissions = corpus_pass(True, submissions)
        measured_finish = time.perf_counter_ns()
        summary = empty_record("ttnn-reference", "measured_phase")
        summary.update({
            "batch": arguments.batch,
            "iteration": submissions,
            "query_count": arguments.iters * query_count,
            "end_to_end_us": elapsed_us(measured_start, measured_finish),
            "peak_host_rss_bytes": peak_host_rss_bytes(),
        })
        writer.writerow(summary)
    finally:
        searcher.close()


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("queries", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--packed-bank", type=Path,
        help="validated RAGTTB01 bank produced by pack_tt_bank.py",
    )
    parser.add_argument(
        "--tracy-signposts", action="store_true",
        help="emit optional Tenstorrent Tracy signposts (diagnostic runs only)",
    )
    arguments = parser.parse_args(argv)
    if arguments.backend != "ttnn-reference":
        parser.error("backend must be ttnn-reference")
    if arguments.precision != "bf16":
        parser.error("ttnn-reference supports only bf16")
    if arguments.batch <= 0 or arguments.warmup < 0 or arguments.iters <= 0:
        parser.error("batch and iters must be positive; warmup must be non-negative")
    if arguments.device < 0:
        parser.error("device must be non-negative")
    if arguments.packed_bank is not None and not arguments.packed_bank.is_file():
        parser.error("packed bank does not exist")
    if not math.isfinite(arguments.threshold) or not -1.0 <= arguments.threshold <= 1.0:
        parser.error("threshold must be finite and in [-1, 1]")
    if not arguments.expected.is_file():
        parser.error("expected-result file does not exist")
    return arguments


def main() -> int:
    try:
        run(parse_arguments(), sys.stdout)
        return 0
    except Exception as error:
        print(f"ttnn_bench: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
