"""TT-NN v0.80 reference backend for resident-bank RAG retrieval.

This module intentionally keeps all TT-NN and PyTorch imports inside
``TtnnReferenceSearcher`` so bank parsing and correctness helpers remain
host-testable. The reference path materializes scores and resolves top-1 on the
host. It is a characterization backend, not the final optimized implementation.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

RAGBANK_MAGIC = b"RAGBANK1"
RAGBANK_HEADER = struct.Struct("<8s4I")
SUPPORTED_TTNN_MAJOR_MINOR = (0, 80)


@dataclass(frozen=True)
class Match:
    accepted: bool
    id: int
    score: float


@dataclass(frozen=True)
class SearchTimings:
    cpu_normalization_us: float
    buffer_preparation_us: float
    device_total_us: float
    h2d_bytes: int
    d2h_bytes: int


@dataclass(frozen=True)
class SearchResult:
    matches: list[Match]
    timings: SearchTimings


def parse_major_minor(version: str) -> tuple[int, int]:
    numeric = version.lstrip("v").split("-", 1)[0].split("+", 1)[0]
    pieces = numeric.split(".")
    if len(pieces) < 2:
        raise ValueError(f"cannot parse TT-NN version: {version!r}")
    try:
        return int(pieces[0]), int(pieces[1])
    except ValueError as exc:
        raise ValueError(f"cannot parse TT-NN version: {version!r}") from exc


def require_supported_version(version: str) -> None:
    actual = parse_major_minor(version)
    if actual != SUPPORTED_TTNN_MAJOR_MINOR:
        expected = ".".join(map(str, SUPPORTED_TTNN_MAJOR_MINOR))
        raise RuntimeError(
            f"TT-NN {version} is unsupported by this provisional adapter; "
            f"expected {expected}.x"
        )


def round_up(value: int, multiple: int = 32) -> int:
    if value <= 0 or multiple <= 0:
        raise ValueError("value and multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def load_bank_keys(path: Path) -> tuple[int, int, list[float]]:
    with path.open("rb") as source:
        raw_header = source.read(RAGBANK_HEADER.size)
        if len(raw_header) != RAGBANK_HEADER.size:
            raise ValueError("truncated RAGBANK1 header")
        magic, entries, dimension, visual_tokens, visual_dimension = (
            RAGBANK_HEADER.unpack(raw_header)
        )
        if magic != RAGBANK_MAGIC:
            raise ValueError("input is not a RAGBANK1 file")
        if entries <= 0 or dimension <= 0:
            raise ValueError("bank dimensions must be positive")
        if (visual_tokens, visual_dimension) != (101, 1024):
            raise ValueError("unsupported visual payload shape")
        value_count = entries * dimension
        raw_keys = source.read(value_count * 4)
        if len(raw_keys) != value_count * 4:
            raise ValueError("truncated RAGBANK1 keys")
        keys = list(struct.unpack(f"<{value_count}f", raw_keys))
    if not all(math.isfinite(value) for value in keys):
        raise ValueError("bank contains non-finite keys")
    return entries, dimension, keys


def normalize_queries(
    queries: Sequence[float], query_count: int, dimension: int
) -> list[float]:
    if query_count <= 0 or dimension <= 0:
        raise ValueError("query count and dimension must be positive")
    if len(queries) != query_count * dimension:
        raise ValueError("query buffer has the wrong size")

    normalized: list[float] = []
    for query_index in range(query_count):
        start = query_index * dimension
        row = [float(value) for value in queries[start : start + dimension]]
        if not all(math.isfinite(value) for value in row):
            raise ValueError("query contains non-finite values")
        norm = math.sqrt(sum(value * value for value in row))
        if norm == 0.0:
            raise ValueError("query has zero norm")
        normalized.extend(value / norm for value in row)
    return normalized


def select_matches(
    scores: Sequence[float],
    query_count: int,
    padded_entries: int,
    valid_entries: int,
    threshold: float,
) -> list[Match]:
    if valid_entries <= 0 or valid_entries > padded_entries:
        raise ValueError("invalid valid-entry count")
    if len(scores) != query_count * padded_entries:
        raise ValueError("score buffer has the wrong size")

    matches: list[Match] = []
    for query_index in range(query_count):
        start = query_index * padded_entries
        best_id = 0
        best_score = float(scores[start])
        if not math.isfinite(best_score):
            raise ValueError("score buffer contains a non-finite value")
        for entry in range(1, valid_entries):
            candidate = float(scores[start + entry])
            if not math.isfinite(candidate):
                raise ValueError("score buffer contains a non-finite value")
            # Strict comparison deliberately preserves the lower ID on a tie.
            if candidate > best_score:
                best_id = entry
                best_score = candidate
        matches.append(Match(best_score >= threshold, best_id, best_score))
    return matches


class TtnnReferenceSearcher:
    """Resident-bank BF16 TT-NN reference implementation for v0.80.x."""

    def __init__(self, bank_path: Path, device_id: int = 0):
        try:
            import torch
            import ttnn
        except ImportError as exc:
            raise RuntimeError(
                "TT-NN reference backend requires the pinned TT-NN and PyTorch "
                "packages on a Tenstorrent host"
            ) from exc

        require_supported_version(ttnn.__version__)
        self._torch = torch
        self._ttnn = ttnn
        self.entries, self.dimension, keys = load_bank_keys(bank_path)
        self.padded_entries = round_up(self.entries)
        self.padded_dimension = round_up(self.dimension)
        self._closed = False
        self.device = ttnn.open_device(device_id=device_id)

        # Keep a transposed [1, 1, K, N] BF16 bank resident in device DRAM.
        bank = torch.zeros(
            (1, 1, self.padded_dimension, self.padded_entries),
            dtype=torch.float32,
        )
        source = torch.tensor(keys, dtype=torch.float32).reshape(
            self.entries, self.dimension
        )
        bank[0, 0, : self.dimension, : self.entries] = source.transpose(0, 1)
        self.device_bank = ttnn.from_torch(
            bank,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def close(self) -> None:
        if not self._closed:
            self._ttnn.close_device(self.device)
            self._closed = True

    def __enter__(self) -> "TtnnReferenceSearcher":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def search_batch(
        self,
        queries: Sequence[float],
        query_count: int,
        threshold: float = 0.7,
    ) -> SearchResult:
        if self._closed:
            raise RuntimeError("TT-NN searcher is closed")

        normalize_start = time.perf_counter_ns()
        normalized = normalize_queries(queries, query_count, self.dimension)
        normalize_finish = time.perf_counter_ns()

        prepare_start = time.perf_counter_ns()
        host_queries = self._torch.zeros(
            (1, 1, query_count, self.padded_dimension),
            dtype=self._torch.float32,
        )
        host_queries[0, 0, :, : self.dimension] = self._torch.tensor(
            normalized, dtype=self._torch.float32
        ).reshape(query_count, self.dimension)
        prepare_finish = time.perf_counter_ns()

        device_start = time.perf_counter_ns()
        device_queries = self._ttnn.from_torch(
            host_queries,
            dtype=self._ttnn.bfloat16,
            layout=self._ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self._ttnn.DRAM_MEMORY_CONFIG,
        )
        device_scores = self._ttnn.matmul(device_queries, self.device_bank)
        host_scores = self._ttnn.to_torch(
            self._ttnn.from_device(device_scores, blocking=True)
        )
        device_finish = time.perf_counter_ns()

        flat_scores = host_scores.reshape(-1).tolist()
        matches = select_matches(
            flat_scores,
            query_count,
            self.padded_entries,
            self.entries,
            threshold,
        )
        return SearchResult(
            matches,
            SearchTimings(
                cpu_normalization_us=(normalize_finish - normalize_start) / 1_000,
                buffer_preparation_us=(prepare_finish - prepare_start) / 1_000,
                # This is host-observed synchronized device-pipeline time. It is
                # not presented as profiler-derived kernel time.
                device_total_us=(device_finish - device_start) / 1_000,
                # TT-NN converts both transferred tensors to BF16.
                h2d_bytes=query_count * self.dimension * 2,
                d2h_bytes=query_count * self.padded_entries * 2,
            ),
        )
