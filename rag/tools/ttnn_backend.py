"""TT-NN v0.80 reference backend for resident-bank RAG retrieval.

This module intentionally keeps all TT-NN and PyTorch imports inside
``TtnnReferenceSearcher`` so bank parsing and correctness helpers remain
host-testable. The reference path materializes scores and resolves top-1 on the
host. It is a characterization backend, not the final optimized implementation.
"""

from __future__ import annotations

import math
import hashlib
import importlib.metadata
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

RAGBANK_MAGIC = b"RAGBANK1"
RAGBANK_HEADER = struct.Struct("<8s4I")
TTBANK_MAGIC = b"RAGTTB01"
TTBANK_HEADER = struct.Struct("<8s8I")
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
    persistent_device_bytes: int
    peak_device_bytes: int
    workspace_bytes: int


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


def installed_ttnn_version(ttnn_module) -> str:
    """Return the TT-NN version for wheels that omit ``ttnn.__version__``."""
    module_version = getattr(ttnn_module, "__version__", None)
    if module_version:
        return str(module_version)
    try:
        return importlib.metadata.version("ttnn")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "cannot determine TT-NN version from the module or package metadata"
        ) from exc


def enable_program_cache(ttnn_module, device) -> None:
    """Enable caching across the two public v0.80 Python API variants."""
    module_function = getattr(ttnn_module, "enable_program_cache", None)
    if callable(module_function):
        module_function(device)
        return
    device_method = getattr(device, "enable_program_cache", None)
    if callable(device_method):
        device_method()
        return
    raise RuntimeError("installed TT-NN does not expose program-cache control")


def high_fidelity_matmul_config(ttnn_module, device):
    """Create the Blackhole-validated BF16/FP32-accumulation configuration."""
    initializer = getattr(ttnn_module, "init_device_compute_kernel_config", None)
    math_fidelity = getattr(ttnn_module, "MathFidelity", None)
    if not callable(initializer) or math_fidelity is None:
        raise RuntimeError(
            "installed TT-NN does not expose compute-kernel configuration"
        )
    return initializer(
        device.arch(),
        math_fidelity=math_fidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
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


def bf16_to_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def load_packed_bank(path: Path) -> tuple[int, int, int, int, list[float]]:
    """Load and validate the deterministic RAGTTB01 deployment artifact."""
    raw = path.read_bytes()
    if len(raw) < TTBANK_HEADER.size:
        raise ValueError("truncated RAGTTB01 header")
    (magic, version, entries, dimension, padded_entries, padded_dimension,
     tile, dtype, layout) = TTBANK_HEADER.unpack_from(raw)
    if magic != TTBANK_MAGIC or version != 1:
        raise ValueError("unsupported packed-bank format or version")
    if tile != 32 or dtype != 1 or layout != 1:
        raise ValueError("unsupported packed-bank representation")
    if (padded_entries < entries or padded_dimension < dimension or
            padded_entries % tile or padded_dimension % tile):
        raise ValueError("invalid packed-bank dimensions")
    payload = raw[TTBANK_HEADER.size:]
    expected = padded_entries * padded_dimension * 2
    if len(payload) != expected:
        raise ValueError("packed-bank payload has the wrong size")
    manifest_path = path.with_suffix(path.suffix + ".json")
    if manifest_path.is_file():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError("packed-bank payload hash does not match manifest")
    words = struct.unpack(f"<{len(payload) // 2}H", payload)
    return (entries, dimension, padded_entries, padded_dimension,
            [bf16_to_float(word) for word in words])


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

    def __init__(
        self, bank_path: Path, device_id: int = 0,
        packed_bank_path: Path | None = None, max_batch: int = 128,
        tracy_signposts: bool = False,
    ):
        try:
            import torch
            import ttnn
        except ImportError as exc:
            raise RuntimeError(
                "TT-NN reference backend requires the pinned TT-NN and PyTorch "
                "packages on a Tenstorrent host"
            ) from exc

        require_supported_version(installed_ttnn_version(ttnn))
        self._torch = torch
        self._ttnn = ttnn
        self._tracy_signpost = None
        if tracy_signposts:
            try:
                from tracy import signpost
            except ImportError as exc:
                raise RuntimeError(
                    "Tracy signposts requested, but the TT-Metal tracy module "
                    "is unavailable"
                ) from exc
            self._tracy_signpost = signpost
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")
        if packed_bank_path is None:
            self.entries, self.dimension, keys = load_bank_keys(bank_path)
            self.padded_entries = round_up(self.entries)
            self.padded_dimension = round_up(self.dimension)
            packed_keys = None
        else:
            (self.entries, self.dimension, self.padded_entries,
             self.padded_dimension, packed_keys) = load_packed_bank(packed_bank_path)
            source_entries, source_dimension, _ = load_bank_keys(bank_path)
            if (source_entries, source_dimension) != (self.entries, self.dimension):
                raise ValueError("packed bank does not match source bank dimensions")
            keys = []
        self.max_batch = max_batch
        self._closed = False
        self.device = ttnn.open_device(device_id=device_id)
        enable_program_cache(ttnn, self.device)
        self.compute_kernel_config = high_fidelity_matmul_config(ttnn, self.device)

        # Keep a transposed [1, 1, K, N] BF16 bank resident in device DRAM.
        bank = torch.zeros(
            (1, 1, self.padded_dimension, self.padded_entries),
            dtype=torch.float32,
        )
        if packed_keys is None:
            source = torch.tensor(keys, dtype=torch.float32).reshape(
                self.entries, self.dimension
            )
            bank[0, 0, : self.dimension, : self.entries] = source.transpose(0, 1)
        else:
            bank[0, 0] = torch.tensor(packed_keys, dtype=torch.float32).reshape(
                self.padded_dimension, self.padded_entries
            )
        self.device_bank = ttnn.from_torch(
            bank,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Reuse the largest host workspace. A view is uploaded for each request;
        # TT-NN owns device allocation and program-cache reuse for the operation.
        self._host_queries = torch.zeros(
            (1, 1, max_batch, self.padded_dimension), dtype=torch.float32
        )
        self.persistent_device_bytes = self.padded_dimension * self.padded_entries * 2

    def close(self) -> None:
        if not self._closed:
            self._ttnn.close_device(self.device)
            self._closed = True

    def __enter__(self) -> "TtnnReferenceSearcher":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def _signpost(self, name: str, state: str, query_count: int) -> None:
        if self._tracy_signpost is not None:
            self._tracy_signpost(
                header=f"rag/{name}/{state}",
                message=f"query_count={query_count}",
            )

    # These deliberately small functions provide stable Python host zones when
    # launched with `python -m tracy -p`. They do not synchronize the device and
    # must not be interpreted as device-kernel durations.
    def _normalize_stage(
        self, queries: Sequence[float], query_count: int
    ) -> list[float]:
        return normalize_queries(queries, query_count, self.dimension)

    def _prepare_stage(self, normalized: Sequence[float], query_count: int):
        host_queries = self._host_queries[:, :, :query_count, :]
        host_queries.zero_()
        host_queries[0, 0, :, : self.dimension] = self._torch.tensor(
            normalized, dtype=self._torch.float32
        ).reshape(query_count, self.dimension)
        return host_queries

    def _upload_stage(self, host_queries):
        return self._ttnn.from_torch(
            host_queries,
            dtype=self._ttnn.bfloat16,
            layout=self._ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=self._ttnn.DRAM_MEMORY_CONFIG,
        )

    def _matmul_stage(self, device_queries):
        return self._ttnn.matmul(
            device_queries,
            self.device_bank,
            compute_kernel_config=self.compute_kernel_config,
        )

    def _download_stage(self, device_scores):
        # This blocking download is the request's natural synchronization point.
        return self._ttnn.to_torch(
            self._ttnn.from_device(device_scores, blocking=True)
        )

    def _reduction_stage(
        self, host_scores, query_count: int, threshold: float
    ) -> list[Match]:
        return select_matches(
            host_scores.reshape(-1).tolist(), query_count,
            self.padded_entries, self.entries, threshold,
        )

    def search_batch(
        self,
        queries: Sequence[float],
        query_count: int,
        threshold: float = 0.7,
    ) -> SearchResult:
        if self._closed:
            raise RuntimeError("TT-NN searcher is closed")
        if query_count > self.max_batch:
            raise ValueError(
                f"query_count {query_count} exceeds configured max_batch {self.max_batch}"
            )

        self._signpost("request", "begin", query_count)
        self._signpost("normalize", "begin", query_count)
        normalize_start = time.perf_counter_ns()
        normalized = self._normalize_stage(queries, query_count)
        normalize_finish = time.perf_counter_ns()
        self._signpost("normalize", "end", query_count)

        self._signpost("prepare", "begin", query_count)
        prepare_start = time.perf_counter_ns()
        host_queries = self._prepare_stage(normalized, query_count)
        prepare_finish = time.perf_counter_ns()
        self._signpost("prepare", "end", query_count)

        device_start = time.perf_counter_ns()
        self._signpost("h2d", "begin", query_count)
        device_queries = self._upload_stage(host_queries)
        self._signpost("h2d", "end", query_count)
        self._signpost("matmul", "begin", query_count)
        device_scores = self._matmul_stage(device_queries)
        self._signpost("matmul", "end", query_count)
        self._signpost("d2h_wait", "begin", query_count)
        host_scores = self._download_stage(device_scores)
        self._signpost("d2h_wait", "end", query_count)
        device_finish = time.perf_counter_ns()

        self._signpost("host_reduction", "begin", query_count)
        matches = self._reduction_stage(host_scores, query_count, threshold)
        self._signpost("host_reduction", "end", query_count)
        self._signpost("request", "end", query_count)
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
                persistent_device_bytes=self.persistent_device_bytes,
                peak_device_bytes=(self.persistent_device_bytes +
                                   query_count * self.padded_dimension * 2 +
                                   query_count * self.padded_entries * 2),
                workspace_bytes=(query_count * self.padded_dimension * 2 +
                                 query_count * self.padded_entries * 2),
            ),
        )
