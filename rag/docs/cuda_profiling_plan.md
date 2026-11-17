# CUDA Timing and Profiling Implementation Plan

## Status and purpose

Implementation status: the optional build flags, timed optimized-CUDA search path, raw
schema, hardware aggregation, NVTX ranges, and Nsight workflow described below are
implemented. CPU and CUDA/NVTX compilation are verified. Final device-runtime validation,
instrumentation-overhead measurement, and representative Nsight captures must be performed
on the Jetson benchmark host.

### Profiling decision after branch migration

- [x] Finish the branch migration: keep the Jetson implementation on
  `al/rag-jetson` and move the Tenstorrent implementation to
  `al/rag-tenstorrent`.
- [x] Keep `steady_clock` as the authoritative source for unprofiled application
  P50/P95/P99 latency and throughput.
- [x] Keep CUDA events as the numeric source for NVIDIA device-stage timing.
- [x] Keep NVTX, Nsight Systems, and Nsight Compute for NVIDIA timeline and
  kernel diagnosis. Do not add Tracy to the NVIDIA benchmark path.

The benchmark currently measures host-visible request latency with
`std::chrono::steady_clock`. That remains the authoritative measurement for application
P50/P95/P99 latency, throughput, startup, and comparisons between CPU, CUDA, and future
Tenstorrent backends.

This plan implements the hardware-instrumentation work deferred by
`benchmark_harness_plan.md`. It adds:

- CUDA events for numeric device-stage durations that can be written to CSV and aggregated.
- NVTX ranges that assign application meaning to CPU and CUDA activity.
- A documented Nsight Systems workflow for inspecting CPU/GPU timelines.

These mechanisms complement rather than replace the existing host timer:

| Mechanism | Role | Used for published benchmark statistics |
| --- | --- | --- |
| `steady_clock` | Caller-visible elapsed time | Yes: application latency and throughput |
| CUDA events | CUDA-stream stage durations | Yes: hardware-layer CUDA metrics |
| NVTX | Semantic labels for code regions | No: annotations only |
| Nsight Systems | External CPU/GPU timeline and diagnosis | No: diagnostic evidence |

## Goals

1. Explain the CUDA portion of host-visible vector-retrieval latency.
2. Measure transfer, conversion, similarity, reduction, and total CUDA-stream time without
   synchronizing after every stage.
3. Populate the existing hardware-layer result fields with explicitly defined metrics.
4. Make Nsight traces easy to capture and interpret using stable NVTX ranges.
5. Preserve CPU-only builds and keep all profiling features optional.
6. Preserve retrieval results and the existing application-level timing boundary.

## Non-goals

- Replacing `steady_clock` with CUDA events or Nsight-derived durations.
- Using profiled runs as the source of application P50/P95/P99 results.
- Optimizing the CUDA pipeline while adding instrumentation.
- Adding pinned buffers, asynchronous pipelining, multiple streams, or copy/compute overlap
  in the same change. Those are subsequent experiments informed by the measurements.
- Implementing Tenstorrent profiling or whole-system energy sampling in this pass.
- Adding detailed hardware counters from Nsight Compute.

## Measurement contract

### Application layer

The existing request timestamps in `src/bench.cpp` remain unchanged:

```text
start
  search(batch)
vector_finish
  retrieve_payloads(matches)
response_finish
```

The resulting measurements retain their current meanings:

- `vector_retrieval_us = vector_finish - start`
- `payload_retrieval_us = response_finish - vector_finish`
- `end_to_end_us = response_finish - start`

For CUDA, `vector_retrieval_us` includes CPU normalization, CUDA API overhead, transfers,
device work, synchronization, and host result construction. It measures what the caller
waits for, not kernel time alone.

### CUDA hardware layer

CUDA-event measurements use events recorded into the same stream as the measured work.
They have the following definitions:

| Field | Boundary |
| --- | --- |
| `h2d_us` | Query upload on the CUDA stream |
| `conversion_us` | FP32-to-BF16 query conversion; unavailable for FP32 |
| `gemm_us` | cuBLAS similarity calculation |
| `reduction_us` | Device top-1 selection and thresholding |
| `d2h_us` | Final result download |
| `device_total_us` | First CUDA operation through completion of final result download |
| `h2d_bytes` | Query bytes submitted to the device |
| `d2h_bytes` | Match-result bytes returned to the host |

CPU normalization and buffer-capacity work are measured separately with `steady_clock`,
because CUDA events cannot measure host execution:

- `cpu_normalization_us`
- `buffer_preparation_us`
- `host_result_construction_us`

Host and device substage measurements are diagnostics. Their sum is not required to equal
`vector_retrieval_us`: CUDA API calls, event overhead, synchronization, and clock-domain
differences account for the remaining time.

### Batch semantics

All stage times describe one batch submission. Raw rows may repeat the batch timing for
each query, matching the existing schema, but aggregation must deduplicate by trial and
submission before calculating percentiles.

The report must distinguish:

- batch duration, such as `device_total_us`;
- normalized device time per query, `device_total_us / batch_size`; and
- throughput, which is computed from the complete measured-phase wall time.

## Design

### 1. Optional build features

Add CMake options:

```cmake
option(RAG_ENABLE_NVTX "Enable NVTX profiling annotations" OFF)
option(RAG_ENABLE_CUDA_TIMING "Collect CUDA event timing" OFF)
```

Requirements:

- `RAG_ENABLE_NVTX` requires `RAG_ENABLE_CUDA` and links NVTX through the installed CUDA
  Toolkit.
- `RAG_ENABLE_CUDA_TIMING` requires `RAG_ENABLE_CUDA`.
- CPU-only builds have no CUDA or NVTX header dependency.
- Instrumentation-disabled builds compile the annotation wrapper to no-ops.
- The manifest records whether NVTX and CUDA-event timing were compiled and enabled.

Recommended build modes:

| Mode | NVTX | CUDA events | Purpose |
| --- | ---: | ---: | --- |
| Release benchmark | Off | Off | Authoritative application metrics |
| Hardware measurement | Off | On | Device-stage metrics |
| Nsight capture | On | Off | Timeline diagnosis |

CUDA events may later be enabled with NVTX for correlation, but the first validation
should measure their overhead independently.

### 2. Timing data model

Add a value type in the CUDA-facing search interface:

```cpp
struct SearchTiming {
  double cpu_normalization_us = 0.0;
  double buffer_preparation_us = 0.0;
  double host_result_construction_us = 0.0;
  double h2d_us = 0.0;
  double conversion_us = 0.0;
  double gemm_us = 0.0;
  double reduction_us = 0.0;
  double d2h_us = 0.0;
  double device_total_us = 0.0;
  std::size_t h2d_bytes = 0;
  std::size_t d2h_bytes = 0;
  bool device_timing_available = false;
};
```

Return results and timing together from a benchmark-specific method:

```cpp
struct TimedSearchResult {
  std::vector<Match> matches;
  SearchTiming timing;
};

TimedSearchResult search_batch_timed(...);
```

Keep the production `search_batch()` interface intact and implement it by calling a shared
internal operation. Avoid mutable global or "last timing" state, which would be fragile
under concurrency and easy for callers to misuse.

The benchmark callback should evolve from returning only `std::vector<Match>` to returning
matches plus optional timing. CPU search returns timing with
`device_timing_available == false`.

### 3. CUDA stream and event ownership

`CudaOptimizedSearcher::Impl` owns:

- a cuBLAS handle using the backend's existing default stream;
- reusable event boundaries for each measured device stage; and
- total-start and total-end events.

Create these resources during backend construction and release them using RAII. Do not
create or destroy events inside the measured request loop.

All measured copies, conversion kernels, cuBLAS calls, reduction kernels, and events must
use the same stream. The initial implementation retains the backend's existing default
stream and blocking copies so instrumentation does not also change execution semantics.

Record all stage events first, then wait once at the existing result-availability boundary.
Do not synchronize between stages. Per-stage synchronization would serialize future
overlap and would change the operation being measured.

Introducing an explicit non-blocking stream, pinned buffers, or asynchronous transfers is
an optimization and requires a separate change and benchmark comparison.

### 4. NVTX annotation layer

Add a small C++ RAII wrapper, for example `include/rag/profile.h`, that creates scoped NVTX
ranges when enabled and becomes a no-op otherwise. Application code should not contain
scattered preprocessor conditionals.

Use stable, hierarchical names:

```text
rag/backend_init
rag/first_request
rag/warmup_phase
rag/measured_phase
  rag/request
    rag/vector_retrieval
      rag/normalize_cpu
      rag/ensure_capacity
      rag/h2d
      rag/bf16_conversion
      rag/gemm
      rag/top1_reduction
      rag/d2h
      rag/construct_matches
    rag/payload_retrieval
```

Use categories, colors, or event payloads for backend, precision, batch size, and
submission number. Do not put unique iteration values in range names because unique names
prevent useful aggregation in Nsight Systems.

NVTX ranges annotate host code. They do not become the source of numeric device-stage
durations in the CSV.

### 5. Raw CSV schema

Extend `rag::bench::Record`, `bench_main.cpp`, and `tools/benchmark_common.py` with:

```text
cpu_normalization_us
buffer_preparation_us
host_result_construction_us
h2d_us
conversion_us
gemm_us
reduction_us
d2h_us
device_total_us
h2d_bytes
d2h_bytes
device_timing_available
```

Use empty CSV fields for unavailable measurements. Do not emit `0` for an unmeasured
stage, because zero means that the stage was measured and took no observable time.

Update the raw-schema version in the run manifest so older raw results fail clearly or are
handled explicitly rather than silently misparsed.

### 6. Harness aggregation

Update `tools/bench_harness.py` to calculate device statistics from one row per trial and
submission. Populate `rag_hardware_layer.csv` with:

- device-total P50/P95/P99 per batch;
- device microseconds per query;
- H2D, conversion, GEMM, reduction, and D2H P50/P95/P99;
- transfer bytes per query;
- effective H2D and D2H bandwidth;
- effective similarity GOP/s;
- effective key-read bandwidth; and
- device time as a percentage of host-visible vector-retrieval latency.

Derived metrics must document their formulas. In particular:

```text
device_us_per_query = device_total_us / batch_size

effective_h2d_GB_per_s = h2d_bytes / h2d_seconds / 1e9

similarity_operations =
    2 * bank_entry_count * embedding_dimension * batch_size

effective_GOP_per_s = similarity_operations / gemm_seconds / 1e9

effective_key_GB_per_s =
    key_bytes_examined_per_batch / gemm_seconds / 1e9
```

If a backend does not provide device timing, leave its hardware fields blank. Never infer
device time by subtracting host stages from application latency.

### 7. Nsight Systems workflow

Add a documented profiling command or a small wrapper script that runs a short,
representative workload, for example:

```powershell
nsys profile `
  --trace=cuda,nvtx,cublas,osrt `
  --sample=none `
  --output=results/profiles/cuda-optimized-b32 `
  <rag_bench> <bank> <queries> <expected> `
  --backend cuda-optimized --precision fp32 `
  --batch 32 --warmup 10 --iters 5
```

Confirm the exact command against the Nsight Systems version installed on the benchmark
host. Capture only a short measured region to limit trace size and profiling overhead.

Store `.nsys-rep` files and exported summaries under `results/profiles/`, separate from
authoritative benchmark CSV files. Record the Nsight Systems version, CUDA version, GPU,
driver, power mode, and clock configuration with each capture.

The trace should be inspected for:

- unexpected GPU idle gaps;
- host waits and implicit synchronization;
- allocation inside the measured request;
- copy/kernel ordering;
- cuBLAS and reduction duration;
- transfer/compute overlap or its absence; and
- work accidentally occurring outside the intended NVTX request range.

## Implementation sequence

### Milestone 1: Contract and schema

1. Add the timing value types and benchmark callback result.
2. Extend raw records and CSV parsing with optional timing fields.
3. Update the manifest with a schema version and instrumentation flags.
4. Update CPU/tooling tests before changing CUDA execution.

Exit criteria:

- CPU-only build and tests pass.
- Existing application metrics retain their definitions.
- Missing hardware measurements are represented as unavailable, not zero.

### Milestone 2: Optimized CUDA event timing

1. Add reusable CUDA events to the existing default-stream execution path.
2. Record H2D, conversion, GEMM, reduction, D2H, and total boundaries.
3. Synchronize only at the final result boundary already imposed by result retrieval.
4. Emit timing and byte counts for the optimized CUDA backend.

Exit criteria:

- FP32 and BF16 results still pass the quality gate.
- All available durations are finite and nonnegative.
- No additional synchronization exists between measured device stages.
- CUDA resources are released correctly on normal and exceptional paths.

### Milestone 3: Hardware-layer reporting

1. Deduplicate device samples by trial and submission.
2. Add percentile and derived-metric calculations.
3. Populate the hardware CSV and Markdown report.
4. Document every formula and timing boundary.

Exit criteria:

- Batch duration, per-query normalized time, and throughput are clearly distinguished.
- Hardware fields remain blank for CPU and uninstrumented backends.
- Report rendering and end-to-end tooling tests pass.

### Milestone 4: NVTX and Nsight Systems

1. Add the optional no-op/RAII NVTX wrapper.
2. Annotate setup, warm-up, measured phases, requests, and CUDA substages.
3. Add the profiling command and environment-recording instructions.
4. Capture and inspect representative FP32 and BF16 traces.

Exit criteria:

- Normal builds do not require NVTX.
- Nsight shows stable nested ranges and correctly associated CUDA operations.
- The trace reveals no unintended measured-path allocation or synchronization.

### Milestone 5: Reference backend and portability follow-up

Instrument `CudaSearcher` only if its device-stage numbers remain useful to the final
comparison. Its deliberate full-score D2H transfer makes it structurally different from
the optimized backend, so results must be labeled accordingly.

Define a parallel timing contract for TT-NN/TT-Metal when that backend exists. Preserve
the application-layer boundary across vendors, but use the vendor-appropriate device
profiler rather than attempting to make NVTX a cross-platform hardware timer.

## Verification strategy

### Correctness

- Compare instrumented and uninstrumented IDs, scores, and accept/reject decisions.
- Run the existing FP32 quality gate for every CUDA configuration.
- Run the BF16 agreement and score-error checks.

### Build coverage

- CPU-only, all instrumentation disabled.
- CUDA, instrumentation disabled.
- CUDA with event timing enabled.
- CUDA with NVTX enabled.
- CUDA with both enabled as a compatibility check.

### Timing sanity checks

- Every measured duration is finite and nonnegative.
- `device_total_us` is not required to equal the exact sum of rounded substage fields, but
  large unexplained differences are flagged for inspection.
- Transfer-byte counts agree with batch size, element type, and result structure size.
- There is exactly one device timing sample per batch submission after deduplication.
- No test asserts tight performance thresholds on shared hardware.

### Overhead study

Run the same fixed workload in four modes:

1. release benchmark;
2. NVTX compiled in but not captured;
3. CUDA events enabled;
4. Nsight Systems capture.

Compare host P50/P95/P99 latency and throughput. Report the observed overhead rather than
assuming it is negligible. Application performance tables should use mode 1; device-stage
tables should use mode 3; mode 4 is diagnostic only.

## Expected file changes

| File | Change |
| --- | --- |
| `rag/CMakeLists.txt` | Optional NVTX and CUDA-timing build features |
| `rag/include/rag/search.h` | Timing result types and timed search interface |
| `rag/include/rag/profile.h` | Optional RAII NVTX abstraction |
| `rag/src/search_cuda.cu` | Reusable events, stage annotations and measurements |
| `rag/include/rag/bench.h` | Optional hardware fields in raw benchmark records |
| `rag/src/bench.cpp` | Propagate timed backend results; preserve host boundaries |
| `rag/bench/bench_main.cpp` | Emit extended CSV and phase annotations |
| `rag/tools/benchmark_common.py` | Parse the new versioned raw schema |
| `rag/tools/bench_harness.py` | Aggregate device metrics and derived values |
| `rag/tools/render_report.py` | Render populated hardware results |
| `rag/tests/rag_test.cpp` | CUDA correctness and timing sanity checks |
| `rag/tests/test_benchmark_tools.py` | Schema, deduplication, and reporting tests |
| `rag/docs/development.md` | Build modes and Nsight capture workflow |
| `rag/docs/performance_evaluation.md` | Metric definitions and populated hardware columns |

## Risks and controls

| Risk | Control |
| --- | --- |
| Event synchronization changes execution | Synchronize once at the final result boundary |
| Instrumentation overhead biases application results | Keep authoritative and profiled runs separate |
| Stream ordering makes timings ambiguous | Record events in the same default stream as all measured CUDA work |
| Schema changes break old result parsing | Version the raw schema and fail clearly |
| Missing metrics appear as real zeros | Represent unavailable values with empty fields plus an availability flag |
| Unique NVTX names prevent aggregation | Use stable names and metadata payloads |
| Instrumentation is mixed with optimization | Defer pinned memory, overlap, and pipeline changes |
| Timing tests become flaky | Test invariants and correctness, not strict duration thresholds |

## Definition of done

The work is complete when:

1. Existing application latency and throughput remain measured with the same
   `steady_clock` boundaries.
2. The optimized CUDA backend emits valid device-stage timings and byte counts in an
   explicitly versioned raw schema.
3. The harness produces reproducible hardware-layer metrics from one sample per batch
   submission.
4. CPU-only and instrumentation-disabled builds remain supported.
5. NVTX ranges make request and CUDA stages clear in an Nsight Systems trace.
6. Correctness gates pass with and without instrumentation.
7. Instrumentation overhead is measured and documented.
8. Documentation states which run mode supplies each reported metric.
