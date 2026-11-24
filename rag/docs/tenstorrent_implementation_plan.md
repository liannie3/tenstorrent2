# Tenstorrent p150a RAG Implementation Plan

## Status and objective

Status: in progress. Phase 0 and the TT-NN v0.80 BF16 reference path are complete.
The repository contains deterministic bank packing, a `rag_bench`-compatible
command adapter, strict C++ wire/reduction contracts, and a portable executable
oracle covering device normalization, sharded scoring, deterministic reduction,
gating, and selected-feature gathering. The build has an explicit, pinned-SDK
`RAG_ENABLE_TT_METAL` boundary. The initial single-core reader/compute/writer
program and launcher have compiled and passed an 8-entry correctness smoke test
on the Blackhole p150a. That prototype still materializes and downloads all scores;
multicore local reduction, device global reduction, compact-result transfer, device
normalization, and production benchmarking remain outstanding.

### Progress snapshot (2026-11-23)

| Capability | Status | Evidence / remaining condition |
| --- | --- | --- |
| Pinned p150a toolchain | Complete | TT-Metal v0.80.0, revision `978f93b56e36b68318d11ecf7b101085a81d718e`; upstream single-core matmul passed. |
| TT-NN BF16 reference | Complete for reference use | Resident key bank, program cache, batch adapter, telemetry, and smoke-quality results are present. It intentionally downloads scores and reduces on the host. |
| Portable correctness contracts | Complete | C++ and Python tests specify partitioning, finite-score validation, lower-ID tie breaking, inclusive thresholding, compact wire results, and feature gathering. |
| Single-core TT-Metal scoring | Complete as a prototype | The 8-entry p150a smoke matched ID/gate; score was `0.996094` versus CPU `1.0`. |
| Sharded compact-result oracle | Complete | The host contract reduces per-shard winners and emits the same 12-byte result required from the final kernels. |
| Multicore score sharding | Implemented; awaiting p150a validation | The launcher discovers the runtime grid, partitions score tiles over active cores, and gives each reader/writer a bounded global tile range. It intentionally retains the full-score diagnostic output. |
| Multicore TT-Metal optimized backend | Not complete | Must shard over the runtime grid, reduce locally and globally on device, and avoid a full score tensor/transfer. |
| Fully device-resident numeric retrieval | Not complete | Query normalization and optional selected-feature gathering remain device work. |
| Final comparison campaign | Not started | Requires correctness gates, ten release trials, timing/profile/power runs, and NVIDIA comparison reports. |

The objective is to add a correct, reproducible Tenstorrent backend and then evolve it
into a device-resident retrieval implementation. After initialization, the optimized
path must keep the key bank on the p150a, execute normalization, similarity scoring,
deterministic top-1 selection, and thresholding on the device, and return only the final
match. An optional extension gathers the selected visual features on the device for a
downstream p150a-resident model.

The implementation has two deliberately separate stages:

1. A TT-NN reference backend that establishes integration and numerical behavior.
2. A fused TT-Metalium backend that avoids materializing or transferring the full score
   matrix.

Do not optimize before the TT-NN backend passes the CPU-reference quality gates.

## 1. Non-negotiable behavior

The existing CPU implementation remains the correctness oracle. Every Tenstorrent
backend must preserve these semantics:

- The input is a batch of 2048-element query embeddings.
- Keys are normalized once; each query is normalized before scoring.
- Similarity is the dot product of normalized vectors.
- The greatest score wins.
- Equal scores select the lower bank ID.
- A result is accepted when `score >= threshold`.
- Padded bank entries are never eligible to win.
- Bank loading and backend initialization are outside steady-state request timing.
- The public result remains `{accepted, id, score}`.

Reduced-precision acceptance requires top-1 agreement, gate agreement, and score-error
checks. Score error alone is insufficient.

## 2. Phase 0: inspect and pin the actual p150a stack

Implementation starts on the target host. Do not select APIs from current online
documentation until the installed release is known.

Capture:

```bash
tt-smi
python -c "import ttnn; print(ttnn.__version__); print(ttnn.__file__)"
python -m pip show ttnn
printf 'TT_METAL_HOME=%s\n' "$TT_METAL_HOME"
uname -a
lspci -D -nn -d 1e52:
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
```

For a source installation also capture:

```bash
git -C "$TT_METAL_HOME" rev-parse HEAD
git -C "$TT_METAL_HOME" describe --tags --always --dirty
git -C "$TT_METAL_HOME" status --short
```

Then:

1. Run the device-health tests supplied by that release.
2. Run its TT-NN and multicore matmul examples.
3. Record driver, firmware, TT-Metal, TT-NN, Python, compiler, and OS versions.
4. Query the logical compute grid at runtime. Never hard-code a grid or 140 cores;
   current p150 firmware exposes 120 Tensix cores and harvesting can affect layout.
5. Pin the selected compatible release/commit for development and CI.

Exit criterion: the selected stack can open the p150a and run a known-good matmul.

## 3. Architecture and ownership boundaries

Keep device dependencies behind backend-specific implementations. The portable bank,
CPU search, benchmark runner, and CSV tools must continue to build without TT-Metal.

Proposed interfaces:

```cpp
enum class SearchPrecision {
  fp32,
  bf16,
  bfp8,
};

class TtMetalSearcher {
 public:
  TtMetalSearcher(const Bank&, SearchPrecision, TtSearchMode);
  ~TtMetalSearcher();

  TtMetalSearcher(const TtMetalSearcher&) = delete;
  TtMetalSearcher& operator=(const TtMetalSearcher&) = delete;

  std::vector<Match> search_batch(
      const std::vector<float>& queries,
      std::size_t query_count,
      float threshold);

  TimedSearchResult search_batch_timed(
      const std::vector<float>& queries,
      std::size_t query_count,
      float threshold);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
```

Use PImpl so TT-Metal headers do not leak into portable public headers. Device, queues,
programs, bank buffers, and work buffers are RAII-owned by `Impl`. Destruction must wait
for outstanding work before releasing resources.

Build options should be independent:

```cmake
option(RAG_ENABLE_TT_METAL "Build the Tenstorrent backend" OFF)
option(RAG_ENABLE_TT_PROFILING "Enable Tenstorrent profiling" OFF)
```

CPU-only, CUDA-only, and Tenstorrent-only builds must remain valid.

## 4. Data representation

Preserve three distinct artifacts:

1. `memory.bank`: portable FP32 source bank and payloads.
2. Packed Tenstorrent bank: normalized, padded, tiled device representation.
3. Packing manifest: source SHA-256, entry count, valid count, embedding dimension,
   padded dimensions, layout, data type, packing version, and TT-Metal version.

The first implementation should use BF16 and a tiled layout. Conceptually:

- Query: `[batch, 2048]`.
- Transposed keys: `[2048, padded_entries]`.
- Scores in the TT-NN reference: `[batch, padded_entries]`.

Pad dimensions to the operation's required tile boundaries, normally multiples of 32.
Mask padded IDs explicitly; do not rely on their initialized numeric values. Validate
that packing preserves ID ordering and that reloading a packed bank is bitwise
reproducible.

Bank conversion is an offline/deployment operation and is not included in request
latency. Bank upload and program compilation are reported as startup metrics.

## 5. Milestone 1: TT-NN reference backend

Implementation status: the v0.80 adapter enables the TT-NN program cache, keeps the
bank and a maximum-batch host workspace resident, accepts the deterministic RAGTTB01
packed artifact, validates its manifest payload hash, and reports transfer/workspace
accounting through the common CSV schema. Its `device_total_us` remains a synchronized
host observation; profiler-derived stage fields intentionally remain unavailable.
The Blackhole reference matmul uses HiFi4 with approximation disabled and FP32
destination accumulation; the initial p150a smoke corpus produced matching IDs and
gate decisions with maximum score error `3.48e-5`.

Implement the simplest correct device path supported by the pinned TT-NN release:

```text
host query
  -> normalize and prepare
  -> copy query to device
  -> TT-NN matmul against resident transposed bank
  -> top-1 selection
  -> copy final match to host
```

It is acceptable for the first version to materialize the score tensor. Prefer an
on-device `argmax`/`topk` only after its tie behavior is tested. If the operation does
not guarantee lower-ID tie selection, perform a deterministic correction or temporarily
copy scores for the reference backend. The optimized backend must enforce the rule on
device.

Keep one device open and keep the bank resident across measured requests. Preallocate
buffers for the largest configured batch. Populate the program cache during warmup.

Expose `ttnn-reference` through `rag_bench` using the same CLI and raw CSV contract as
the other backends. If the installed release only exposes a stable Python TT-NN API,
start with a small executable Python adapter that emits the exact benchmark CSV schema;
do not embed Python in `rag_core`. Replace or supplement it with the C++ backend when the
pinned release's supported C++ integration is confirmed.

Exit criteria:

- Fixture and adversarial quality tests pass.
- Batch sizes 1, 2, 4, 8, 16, 32, 64, and 128 work.
- The bank and reusable work buffers remain resident.
- Warm and cold measurements are distinguishable.
- No score-matrix timing is presented as the optimized result.

## 6. Milestone 2: benchmark and measurement integration

Preserve the existing application timing boundary based on host `steady_clock`. Add a
parallel Tenstorrent hardware measurement contract without pretending CUDA events and
TT profiling are identical.

Required logical stages:

- `cpu_normalization_us`
- `buffer_preparation_us`
- `h2d_us`
- `conversion_us`
- `gemm_us` or `dot_product_us`
- `reduction_us`
- `d2h_us`
- `device_total_us`
- `h2d_bytes`
- `d2h_bytes`

If the pinned stack cannot provide a reliable numeric stage duration, leave it
unavailable rather than inserting zero. Record timing source and availability in the
manifest. Update the schema version if fields or meanings change.

Use four run modes:

| Mode | Instrumentation | Purpose |
| --- | --- | --- |
| Release | Off | Authoritative latency and throughput |
| Hardware timing | Minimal device timing | Hardware-layer stage metrics |
| Tracy/device profiler | On | Timeline, per-core, and NoC diagnosis |
| Power sampling | External telemetry | Energy, temperature, and throttling |

Profiler runs are not the source of published P50/P95/P99 request latency. Add stable
signposts around bank upload, request, H2D, compute, reduction, D2H, and result
construction. Deduplicate device timing once per batch submission just as the CUDA
aggregation does.

The TT-NN adapter now provides optional `--tracy-signposts` bracketing request,
normalization, preparation, H2D submission, matmul submission, blocking D2H wait, and
host reduction. Running it with `python -m tracy -p` also exposes stable Python helper
functions as host zones. No intermediate device synchronization is introduced.

## 7. Milestone 3: fused TT-Metalium retrieval

Implementation status: `tools/tt_metal_model.py` and the portable C++
`rag/tt_metal_contract.h` freeze the runtime partitioning, deterministic reduction,
and 12-byte device/host result contract. Their adversarial host tests cover uneven
partitions, excess workers, ties, padding, threshold equality, non-finite values,
wire-result validation, normalization, fused scoring, and selected-feature gathering.
The single-core Metalium reader/compute/writer programs and launcher have been built
and validated on the pinned p150a stack. This is only a scoring prototype: it writes
the complete score matrix to DRAM, downloads it, and reduces it on the host. The
portable launcher now routes those scores through the sharded local/global reduction
and compact wire-result oracle, establishing the seam that the next device kernels
must replace. The optimized backend must not be reported as complete until the same
behavior executes on device and only the compact result crosses PCIe.

The optimized backend must eliminate the full score transfer and should eliminate the
score tensor in GDDR entirely.

```text
query upload/normalization
          |
          v
query replicate or multicast
          |
          v
GDDR bank shards -> Tensix reader -> circular buffers
                                      |
                                      v
                              dot-product compute
                                      |
                                      v
                          per-core {score, id}
                                      |
                                      v
                       deterministic global reduction
                                      |
                                      v
                         threshold + compact result
                                      |
                                      v
                            one D2H result/query
```

Kernel responsibilities:

- Reader/data-movement kernel streams a core's key shard from GDDR into L1 circular
  buffers and obtains the query through replication or multicast.
- Compute kernel accumulates dot products and maintains local best candidates.
- Writer/data-movement kernel publishes local winners or the final compact result.
- A reduction stage compares `(score, id)` pairs using:

```cpp
candidate.score > best.score ||
    (candidate.score == best.score && candidate.id < best.id)
```

Query runtime grid dimensions through `compute_with_storage_grid_size()` and use
runtime work splitting. Do not assume a particular physical or logical grid. Assign
contiguous ID ranges so tie handling and result reconstruction are simple.

Use double-buffered circular buffers where measurement shows useful overlap between
GDDR reads and compute. Measure before adding multicast, multiple queues, or additional
pipeline complexity.

Exit criteria:

- No complete score tensor crosses PCIe.
- The normal path does not materialize the complete score tensor in GDDR.
- Only one compact result per query is downloaded.
- All quality gates pass.
- Production-corpus latency or throughput improves over `ttnn-reference`.

## 8. Milestone 4: fully device-resident numeric retrieval

Move query normalization and thresholding into the device program. Keep normalized keys,
programs, and work buffers resident. Then optionally keep the visual-feature payload as
a device tensor and gather the selected `[101, 1024]` features on device.

Two supported outputs should remain explicit:

1. Compact match: `{accepted, id, score}` for a host-managed downstream pipeline.
2. Compact match plus selected numeric features for a p150a-resident downstream model.

Do not move variable-length description/string processing into Metalium merely to claim
that the application is fully on device. Store strings as host-side payloads unless an
end-to-end device model has a concrete need for tokenized device-resident data.

End-to-end query encoding or language-model generation is a later project. It must not
block completion of device-resident retrieval.

## 9. Correctness and test plan

Add tests at three levels.

### Host-only tests

- Packing manifest and hash validation.
- Padding and valid-ID masking.
- Work partitioning for uneven entry and core counts.
- Deterministic `(score, id)` comparison.
- CSV schema and unavailable timing behavior.
- Backend selection when TT-Metal is not compiled.

### Device correctness tests

- Existing deterministic fixture.
- Random banks compared with CPU FP32.
- Exact ties with the lower ID expected.
- Near ties within BF16 rounding distance.
- Scores immediately below, equal to, and above the threshold.
- Non-multiple-of-32 entry counts and batch sizes.
- Empty/invalid inputs rejected by contract.
- Repeated requests to detect stale buffers or queue-ordering errors.

### Quality gates

- Top-1 agreement: target 100% for the approved corpus.
- Gate agreement: 100%.
- Mean and maximum score error: recorded, with the initial BF16 maximum tolerance no
  looser than the existing `5e-3` unless evidence justifies a reviewed change.
- Optimized results exactly match the TT-NN backend's IDs and decisions on adversarial
  tie/threshold tests.

Device tests should be labeled and skipped clearly when no p150a is available; they must
not masquerade as passing host tests.

## 10. CPU affinity, NUMA, and benchmark controls

CPU pinning is not required for implementation or correctness. For final measurements,
control host placement because the p150a is a PCIe device and host scheduling can affect
small-batch latency.

Find the device and its NUMA node:

```bash
lspci -D -d 1e52:
cat /sys/bus/pci/devices/0000:BB:DD.F/numa_node
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
```

First use NUMA-local CPU and memory placement:

```bash
numactl --cpunodebind=<node> --membind=<node> <benchmark-command>
```

Only add `taskset` after measuring scheduler variance. Give TT-NN/Metal runtime worker
threads a set of physical cores; do not pin the entire process to one CPU. Avoid mixing
SMT siblings into the set unless measurement shows a benefit.

Run and report an affinity experiment with:

1. Default scheduler placement.
2. NUMA-local placement.
3. NUMA-local placement plus a fixed physical-core set.

Adopt the least restrictive configuration that materially reduces variance without
reducing throughput. Record the CPU mask, NUMA policy, governor, background-load policy,
and whether SMT is enabled. Apply equivalent host controls to NVIDIA and Tenstorrent
runs.

## 11. Benchmark matrix and reporting

For release results use:

- Backends: CPU, CUDA optimized, TT-NN reference, TT-Metal optimized.
- Batch sizes: 1, 2, 4, 8, 16, 32, 64, 128.
- Precisions: BF16 first; BFP8 only after BF16 sign-off.
- Warmup: enough to populate program caches, recorded explicitly.
- Measured iterations: 100 corpus passes unless runtime demands a documented adjustment.
- Independent trials: 10 fresh processes.
- Bank sizes: fixture, intermediate, and production corpus.

Report:

- Bank load, device initialization, compilation/program-cache warmup, first request,
  and cold start separately.
- P50/P95/P99 host-visible latency and run-to-run throughput variation.
- Device-stage times and bytes only from the hardware-timing run.
- Persistent bank memory, workspace, and any score-buffer memory.
- Top-1/gate agreement and score error.
- Power, temperature, clocks, and throttling state when available.
- TT-Metal/TT-NN, driver, firmware, logical grid, PCIe, NUMA, and CPU-affinity metadata.

The comparison must use the same source bank hash, queries, threshold, batch semantics,
warmup policy, and application timing boundaries on NVIDIA and Tenstorrent.

## 12. Performance experiments after correctness

Change one factor at a time and retain the baseline:

1. Number and shape of participating Tensix cores.
2. Bank shard size and contiguous versus distributed placement.
3. Query replication versus NoC multicast.
4. Circular-buffer depth and single versus double buffering.
5. BF16 versus BFP8 storage/compute.
6. Materialized score matrix versus fused local reduction.
7. One versus multiple command queues and transfer/compute overlap.
8. Device feature gather versus host bank lookup.

Measure device time, host latency, bandwidth, slowest-core completion, NoC activity,
workspace, quality, and energy. Using all available cores is not automatically optimal;
stop scaling when GDDR, NoC, submission overhead, or reduction dominates.

## 13. Expected repository changes

| File | Change |
| --- | --- |
| `CMakeLists.txt` | Optional TT-Metal backend and profiling features |
| `include/rag/search.h` | Tenstorrent backend declaration and generalized precision |
| `src/search_tt.cpp` | RAII host implementation and TT-NN/Metal dispatch |
| `src/tt/kernels/*` | Reader, dot-product, reduction, and writer kernels |
| `bench/bench_main.cpp` | Tenstorrent backend selection and instrumentation flags |
| `include/rag/bench.h` | Timing-source and Tenstorrent availability metadata |
| `src/bench.cpp` | Timed Tenstorrent callback integration |
| `tools/pack_tt_bank.py` | Reproducible packed bank and manifest generation |
| `tools/benchmark_common.py` | Versioned schema parsing |
| `tools/bench_harness.py` | Backend orchestration and device-metric aggregation |
| `tools/render_report.py` | Tenstorrent environment and hardware results |
| `tests/rag_test.cpp` | Portable comparator/partition tests and device tests |
| `tests/test_benchmark_tools.py` | Schema, manifest, timing, and report tests |
| `docs/development.md` | Build, execution, profiling, and NUMA instructions |

Exact source structure may change to match the pinned TT release, but the boundaries and
test responsibilities should remain.

## 14. Risks and controls

| Risk | Control |
| --- | --- |
| Installed APIs differ from current docs | Inventory and pin the actual stack first |
| Firmware changes visible core layout | Query the runtime grid; never hard-code it |
| TT-NN C++ API is not stable/supported | Begin with an executable Python reference adapter |
| Padding changes the winner | Explicit valid-ID mask and adversarial tests |
| Reduced precision changes ties or threshold decisions | ID/gate quality gates and boundary corpus |
| Program compilation contaminates latency | Warm program cache and report cold start separately |
| Async queues make timing incorrect | Explicit final synchronization and documented boundaries |
| Profiling changes performance | Separate release, timing, profiler, and power runs |
| Pinning starves runtime helper threads | Use a NUMA-local physical-core set, not one CPU |
| Full score tensor dominates memory traffic | Fused local and global top-1 reduction |
| Device code becomes coupled to application payloads | Keep numeric retrieval and payload policy separate |

## 15. Delivery sequence

1. [x] Target-host inventory and pinned stack record.
2. [x] Backend CLI/build skeleton plus host-only tests.
3. [x] Bank packer and manifest.
4. [x] TT-NN BF16 reference with resident bank.
5. [x] Initial quality-gate and benchmark integration.
6. [x] Separate timing and Tracy profiler workflow.
7. [x] Single-core Metalium dot-product prototype.
8. [~] Multicore score sharding implemented; p150a validation and local-winner fusion remain.
9. [ ] Deterministic global reduction, thresholding, and compact D2H result.
10. [ ] Persistent bank/program lifecycle and batch-capable benchmark adapter.
11. [ ] Device normalization and optional visual-feature gather.
12. [ ] NUMA/affinity experiment and final benchmark campaign.
13. [ ] Documentation, reproducibility audit, and comparison report.

### Execution plan from the current checkpoint

Each increment must preserve a passing host-only build and must be separately
validated on the pinned p150a before the next optimization is layered on top.

1. **Freeze the reduction seam.** Use the portable sharded-score oracle from the
   single-core launcher; test worker-count invariance, exact ties, rejected winners,
   invalid scores, and the 12-byte result. This is implemented in the current tree.
2. **Create a multicore scoring prototype.** Discover the runtime compute/storage
   grid, call `partition_entries`, assign contiguous entry ranges, and run one program
   per active core. Initially permit one score shard per core in DRAM so work splitting,
   tile offsets, and numerical behavior can be validated independently. The code is
   implemented; compilation and correctness validation on the pinned p150a are next.
3. **Fuse per-core local selection.** Replace each score shard with one local
   `{score, id, valid}` candidate. Exclude padded columns explicitly. Validate each
   local winner against the portable oracle for uneven partitions and bank sizes
   `1, 31, 32, 33, 3400`.
4. **Add deterministic global reduction.** Reduce candidates on device using higher
   finite score then lower ID, apply the inclusive threshold, and write one
   `CompactResult` per query. Confirm that D2H bytes are 12 per query and that no
   complete score tensor exists on the normal path.
5. **Turn the prototype into a backend.** Introduce a RAII searcher that owns the mesh
   device, resident packed bank, compiled programs, queues, and reusable batch buffers.
   Add a `tt-metal-optimized` command adapter with the existing benchmark CSV contract.
6. **Move normalization to device.** Add norm reduction, reciprocal square root, and
   normalized query distribution. Retain a temporary diagnostic mode that accepts a
   host-normalized query for differential testing, but exclude it from release results.
7. **Complete optional feature handoff.** If the downstream model will remain on the
   p150a, gather the selected numeric feature tensor on device. Otherwise record this
   explicitly as follow-up and retain host-side strings/payload lookup.
8. **Qualify correctness and performance.** Run fixture, adversarial, randomized, and
   production-corpus comparisons; then run release, device-timing, Tracy, power, and
   NUMA/affinity campaigns separately. Require a measured production-corpus benefit
   over `ttnn-reference` before calling the optimized backend complete.

Each step must leave the CPU/CUDA paths buildable and tested. Do not combine the TT-NN
reference and fused-kernel optimization into one unreviewable change.

## 16. Definition of done

The Tenstorrent implementation is complete when:

1. A clean build without TT-Metal remains supported.
2. The pinned p150a stack and runtime grid are recorded in every result manifest.
3. `ttnn-reference` passes the CPU quality gates with a resident bank.
4. `tt-metal-optimized` normalizes, scores, reduces, and thresholds on device.
5. The optimized path returns only one compact result per query and does not materialize
   or transfer a complete score tensor.
6. Exact ties and threshold-boundary cases are deterministic.
7. Release, device-timing, Tracy, and power runs remain separate.
8. Ten independent release trials produce application and hardware reports with the same
   workload contract as the NVIDIA comparison.
9. Instrumentation overhead and CPU-affinity effects are measured and documented.
10. The optimized backend demonstrates a measured benefit over the TT-NN reference on
    the production corpus without failing quality gates.
11. Optional device-resident feature gathering is either implemented and tested or
    explicitly recorded as follow-up work.

## References

- [Blackhole p150 specifications](https://docs.tenstorrent.com/aibs/blackhole/)
- [TT-Metalium documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/)
- [TT-NN documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/)
- [Single-core matmul and Tensix architecture](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab1/lab1.html)
- [Multicore work splitting](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab2/lab2.html)
- [Tracy profiler](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html)
