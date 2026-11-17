# RAG Benchmarking Harness — Plan

## Implementation status

Milestones 1-6 and the service-stage instrumentation pass are implemented. The repository
now contains the deterministic synthetic fixture, persistent in-process `rag_bench`
runner, native batch API, quality-gated Python orchestration, raw and summarized CSV
outputs, report rendering, and end-to-end smoke tests. It measures host-facing end-to-end,
vector-retrieval, and conditional payload-retrieval latency; separates accepted and
rejected batch-1 paths; and calculates throughput from the complete measured-phase wall
clock. The CUDA build contains both the original reference backend and an optimized cuBLAS
backend with device-side top-1/gating and FP32 or BF16 inputs.

As scoped below, power-sampler integration, hardware-substage/device-event timing,
peak-memory sampling, bank-size sweeps, concurrency, and TT-NN/TT-Metal backends remain
follow-on work for the target hardware/runtime integration pass. The bank-size follow-on
will use deterministic synthetic banks rather than wait for a larger real corpus. CUDA
execution tests are conditional: they run automatically when CTest sees a CUDA device and
otherwise leave CPU and toolchain tests active.

## Context

`performance_evaluation.md` specifies exactly what the RAG subsystem comparison needs:
batch-1 P50/P95/P99 latency, sustained queries/s, cold-start/bank-load time, quality-gate
agreement vs. the FP32 CPU reference, and (later) whole-SUT energy per query — all under a
committed, deterministic query corpus run identically on every backend. None of that exists
today. `rag_query` runs exactly one query per process invocation and prints a single line;
there is no timing loop, no percentile computation, no corpus, and no correctness-gating
step. Before any cross-platform number is meaningful, this harness has to exist and has to
be pluggable, because two more backends (`rag_cuda_optimized`, and eventually
`rag_ttnn`/`rag_tt_metal`) are coming — the harness must not be rewritten each time a new
backend is added.

The CUDA optimization list (persistent bank, device-side top-1, cuBLAS/batched similarity,
BF16 path) is being implemented separately. This plan covers the harness only. Per
discussion:

- **No bank-size sweep in v1.** The 25/50/75/90%-of-memory-budget workload matrix is
  deferred from the initial harness; v1 benchmarks whatever single bank is configured.
  The next phase will generate deterministic 10-, 100-, 3,400-, and 10,000-entry banks,
  then increase the entry count until a platform reaches its memory limit.
- **Service-stage timing is implemented.** The harness times vector retrieval and
  conditional payload retrieval without forcing fused hardware operations apart. Query
  placement, similarity, top-1/gating, result transfer, and synchronization remain
  profiler/device-event diagnostics for the hardware integration pass.
- **Energy is a pluggable interface only.** No tegrastats/tt-smi wiring or testing now —
  just a hook shaped so it can be filled in once run on Jetson/p150a hardware.

One gap this plan has to account for: **no real bank or query corpus is committed to the
repo yet** (only `rag/tests/test_builder.py`'s tiny in-memory fixture exists). The harness
is built and validated against a small deterministic synthetic bank, with the real
3,400-entry bank dropped in later via the same `--bank` path argument — no code change
required when it arrives.

## Architecture

Two new layers, matching the existing split between the C++ core and the Python tooling
(`tools/build_bank.py`):

1. **`rag_bench` (C++, new binary, reuses `rag_core`)** — loads the bank once, loads a
   committed query corpus once, runs a warmup phase then a measured phase in-process (no
   per-query process spawn, so process-launch overhead never contaminates microsecond-scale
   CPU/CUDA latencies), and emits one raw record per query. It is backend-selectable the
   same way `rag_query` is today (`--cuda` flag), but structured so a future backend is one
   more `if`/enum branch, not a new binary.
2. **Python orchestration (`tools/`)** — generates the deterministic corpus, drives
   `rag_bench` across the configurations in the workload matrix (backend × batch × cold/warm),
   computes percentiles/throughput from the raw records, runs the quality-gate comparison,
   and renders results into the Markdown tables already present in
   `performance_evaluation.md`. This mirrors `build_bank.py`'s existing pattern of
   stdlib-only Python tools with no new dependency.

### New files

- `bench/bench_main.cpp` — the `rag_bench` entry point.
- `include/rag/bench.h` + `src/bench.cpp` — shared benchmark-loop logic (warmup
  count, measured count, per-query timing, record struct) used by `bench_main.cpp`, kept out
  of `rag_core` since it's benchmark-only, not part of the retrieval library's public
  contract.
- `tools/generate_corpus.py` — builds a deterministic synthetic bank + matching query
  set from a fixed seed, and (separately) can point at a real bank once one exists; also
  runs `rag_query`/CPU search once per query to record the FP32 "expected" result
  (`top1_id`, `accepted`, `score`) as the reproducibility baseline.
- `bench_data/` — committed output of the above for the synthetic fixture: `bank.bin`,
  `queries.f32` (concatenated 2048-float vectors), `expected.json`. Real-bank runs point
  `--bank`/`--queries` elsewhere via CLI flags; nothing about the corpus format is
  bank-specific.
- `tools/bench_harness.py` — orchestrator. For each (backend, batch, mode) triple in the
  workload matrix: invokes `rag_bench`, parses its raw CSV/JSONL output, computes P50/P95/P99
  and queries/s, and writes one row per configuration to `results/rag_application_layer.csv`
  and `results/rag_hardware_layer.csv` (column names taken directly from the two result
  tables in `performance_evaluation.md`).
- `tools/quality_gate.py` — runs the full corpus through CPU and every other configured
  backend, computes top-1 ID agreement %, gate (accept/reject) agreement %, and score error
  (mean/max abs diff) vs. FP32 CPU, and writes `results/rag_quality_gate.csv`. Per the doc's
  decision rules, `bench_harness.py` refuses to report a backend's perf numbers if this gate
  fails for that backend (configurable agreement threshold, default from the doc's "same
  minimum top-1 and gate-agreement requirements").
- `tools/render_report.py` — reads the three CSVs above and fills the existing Markdown
  tables in `performance_evaluation.md` (Application Layer results, Hardware Layer results)
  in place, so results land in the doc instead of living only as loose CSVs.
- `results/` — generated run output, gitignored (raw data is reproducible from the committed
  corpus + a fixed seed; only the CSVs a human wants to diff get committed intentionally).

### `rag_bench` behavior

- CLI: `rag_bench BANK QUERIES EXPECTED [--cuda] [--batch N] [--warmup N] [--iters N]`.
- Load the bank once (reusing `rag::Bank`); construct the backend once (`CudaSearcher`
  persists across all queries — this is the harness's main answer to "construct
  `CudaSearcher` per CLI query is unfair," independent of whatever persistent-state work
  happens inside `rag_cuda_optimized` itself).
- Cold-start timing: measure process-start-to-first-result once, separately from the warm
  loop, and report it as its own record — matches the doc's "cold-start time" and
  "bank-load time" metrics.
- Warm loop: run `warmup` iterations (discarded), then `iters` measured iterations over the
  query set, timing each with `std::chrono::steady_clock` around the exact same
  submission-to-payload-availability boundary `rag_query` documents today. Batch size N means
  N queries submitted together where the backend supports it; batch 1 is always run.
- Output: one line per measured query with submission/query identity, end-to-end time,
  vector-retrieval time, payload-retrieval time, returned payload bytes, and match fields.
  Setup and measured-phase summary rows use the same CSV schema. Raw output is retained
  per backend and batch so the summaries remain auditable.
- Determinism: same query order every run, no dataset shuffling, so different backends and
  different days are comparable (matches the doc's reproducibility rules).

### Quality gate before any perf claim

`quality_gate.py` is not optional plumbing — the doc is explicit that a result is invalid if
it misses the quality gate, and that raw comparisons must reject anything that does. It runs
before `bench_harness.py` publishes a number for a given backend, using the same corpus and
`expected.json` that `rag_bench` already consumes, so there's exactly one corpus/expected-result
source of truth for both correctness and performance runs.

### Energy hook (interface only)

`bench_harness.py` accepts an optional `--power-sampler <command>` argument. If given, it
launches that command as a subprocess for the duration of each measured run and expects it to
emit periodic `timestamp,watts` lines on stdout; the harness averages them into a
whole-SUT-joules/query column. If omitted (the default, and the only mode actually exercised
now), the energy columns are left blank. No tegrastats or tt-smi integration is written or
tested in this pass — this only defines the shape a later integration plugs into.

## CMake changes

Add to `rag/CMakeLists.txt`:

```cmake
add_library(rag_bench_lib src/bench.cpp)
target_link_libraries(rag_bench_lib PUBLIC rag_core)

add_executable(rag_bench bench/bench_main.cpp)
target_link_libraries(rag_bench PRIVATE rag_bench_lib)
```

Placed after the existing `rag_query` target, following the same `target_link_libraries`
pattern already used there.

## Milestones

1. `generate_corpus.py` — deterministic synthetic bank + query set + FP32 expected results
   committed under `bench_data/`.
2. `rag_bench` C++ binary — cold-start record, warm warmup/measured loop, raw per-query
   output, backend-selectable, batch-1 only to start.
3. `bench_harness.py` — drives `rag_bench` across backend × mode, computes percentiles and
   throughput, writes the Application/Hardware Layer CSVs.
4. `quality_gate.py` — top-1/gate agreement + score error, gates step 3's output.
5. Batch-N support in `rag_bench` + harness, once a backend (`rag_cuda_optimized`) actually
   supports batched submission.
6. `render_report.py` — fills the Markdown tables in `performance_evaluation.md` from the
   CSVs.
7. (Deferred, documented only) energy sampler wiring; bank-size sweep; stage-level tracing.

### Planned synthetic bank-size sweep

The controlled sweep uses the same fixed seed, normalized FP32 keys, 2,048-element
dimension, query corpus, planted matches, and gate-boundary cases at every bank size. Run
the same fixed entry counts on both platforms; report platform-specific 25%, 50%, 75%, and
90% memory-capacity runs separately.

Maintain two experiment paths:

- **Retrieval-only scaling:** avoid materializing full visual payloads so the sweep isolates
  similarity-search latency, throughput, and resident key memory.
- **Full-bank scaling:** retain each entry's description and 101 by 1,024 FP32 visual
  features to measure payload access, peak memory, startup time, and maximum capacity.

A full entry occupies about 412 KiB before description and runtime overhead, so 10,000
entries require more than 4.2 GB before runtime allocations. Every result must record entry
count, bank-file bytes, host/device memory, latency percentiles, sustained queries/s, and
bank-load/device-placement time.

## Verification

- Build both variants and run the existing test suite unaffected:
  `cmake -S . -B build-cpu -DRAG_ENABLE_CUDA=OFF && cmake --build build-cpu && ctest --test-dir build-cpu`
  and the `-cuda` equivalent.
- `python tools/generate_corpus.py` produces `bench_data/bank.bin`,
  `queries.f32`, `expected.json` deterministically (re-run twice, diff outputs — must be
  byte-identical given the fixed seed).
- Run `rag_bench` directly against the synthetic fixture for both `cpu` and `--cuda`;
  confirm cold-start record + N measured records appear, and that CPU's own reported
  `accepted/id/score` matches `expected.json` exactly (CPU is the reference, so this is a
  harness self-check, not a conformance test).
- Run `quality_gate.py` against `cpu` vs. `--cuda`; confirm 100% top-1/gate agreement on the
  synthetic fixture (both use the same FP32 math today — this is a harness correctness check,
  not yet a real cross-precision conformance result).
- Run `bench_harness.py` end-to-end; confirm `results/rag_application_layer.csv` has
  batch-1 P50/P95/P99 and queries/s populated for both backends, and that
  `render_report.py` fills the corresponding row in `performance_evaluation.md` without
  disturbing the rest of the file.
- Once the CUDA-optimization work lands `rag_cuda_optimized`, re-point `bench_harness.py`'s
  backend list at it with no code changes beyond adding its name to the config list — this
  is the harness's main design test: a new backend must be config, not a rewrite.
