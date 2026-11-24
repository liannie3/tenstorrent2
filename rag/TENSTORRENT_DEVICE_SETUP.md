# Tenstorrent p150a RAG Device Setup

This guide prepares a Linux host with a Blackhole p150a for the TT-NN reference backend,
the later fused TT-Metalium backend, thermal telemetry, and profiling. The provisional
development target is TT-Metal/TT-NN **v0.80.0**. Validate the actual installed stack
before upgrading or replacing anything.

The implementation status and milestones are tracked in
[`rag/docs/tenstorrent_implementation_plan.md`](rag/docs/tenstorrent_implementation_plan.md).

## 1. Validate hardware and installed software

Run these commands on the p150a host and save their complete output:

```bash
curl -fsS "${TT_TELEMETRY_ENDPOINT:-http://127.0.0.1:8080/metrics}" \
  | grep -E '^(tt_asic_temperature_celsius|tt_tdp_w|tt_ai_clock_mhz)'
tt-smi -s  # retained only as an inventory and compatibility check
lspci -D -nn -d 1e52:
uname -a
python3 --version
python3 -c "import ttnn; print(ttnn.__version__); print(ttnn.__file__)"
python3 -m pip show ttnn
printf 'TT_METAL_HOME=%s\n' "$TT_METAL_HOME"
```

For a source installation:

```bash
git -C "$TT_METAL_HOME" rev-parse HEAD
git -C "$TT_METAL_HOME" describe --tags --always --dirty
git -C "$TT_METAL_HOME" status --short
```

Run the device-health and matmul examples shipped with that exact release. Do not install
TT-NN independently of its compatible TT-Metal, firmware, driver, and TT-SMI stack.

Current p150 firmware exposes 120 Tensix cores. Software must query the logical compute
grid at runtime and must not hard-code a core count or grid dimensions.

## 2. Provisional software target

Until the actual host inventory is available, development assumes:

```text
TT-Metal/TT-NN: v0.80.x
Hardware:       Blackhole p150a
Initial dtype:  BF16
Reference API:  TT-NN Python
Optimized API:  TT-Metalium C++ and device kernels
```

The current adapter rejects a different TT-NN major/minor version deliberately. If the
device uses another compatible release, update and test the narrow adapter boundary; do
not silently ignore the mismatch.

## 3. Clone and create the project environment

```bash
git clone https://github.com/liannie3/tenstorrent2.git
cd tenstorrent2
git switch al/rag-jetson
cd rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Activate the environment or container supplied by the selected TT-Metal release before
running TT-NN. Confirm that `import ttnn` succeeds from the project shell.

## 4. Build and test the portable code

The CPU baseline and host-only Tenstorrent packing/contract tests do not require device
headers:

```bash
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

This validates the portable bank, deterministic BF16 packer, TT-NN adapter contract,
telemetry parsers, and benchmark tooling.

## 5. Create the Tenstorrent packed bank

First create or select the portable source bank:

```bash
python tools/build_bank.py entries.jsonl memory.bank \
  --metadata bank_metadata.json
```

Then create the deterministic, transposed, tile-aligned BF16 artifact:

```bash
python tools/pack_tt_bank.py memory.bank memory.ttbank \
  --tt-metal-version v0.80.0
```

Keep these artifacts distinct:

- `memory.bank`: portable normalized FP32 keys and payloads.
- `memory.ttbank`: transposed and 32-aligned BF16 key matrix.
- `memory.ttbank.json`: source hash, valid/padded shapes, layout, data type, payload hash,
  packing version, and TT-Metal version.

Padded entries are storage only and must always be excluded from top-1 selection.

## 6. TT-NN reference smoke test

The repository contains the resident-bank implementation in
`rag/tools/ttnn_backend.py`, a `rag_bench`-compatible adapter in
`rag/tools/ttnn_bench.py`, and host-only contract tests. Validate the module and
installed API directly:

```bash
python tests/test_ttnn_backend.py
python -c "from pathlib import Path; from tools.ttnn_backend import TtnnReferenceSearcher; s=TtnnReferenceSearcher(Path('bench_data/bank.bin')); print(s.entries, s.dimension, s.padded_entries); s.close()"
```

The reference backend keeps a transposed BF16 bank resident, performs TT-NN matmul, and
currently returns the score matrix for deterministic host-side top-1. It is a correctness
baseline, not the final performance design.

Run the adapter through the normal harness with:

```bash
python tools/bench_harness.py \
  --rag-bench build/rag_bench \
  --ttnn-bench tools/ttnn_bench.py \
  --bank bench_data/bank.bin \
  --queries bench_data/queries.f32 \
  --expected bench_data/expected.json \
  --backends ttnn-reference:bf16 \
  --batches 1 --warmup 1 --iters 1 --trials 1 \
  --results results/smoke-tenstorrent
```

Its required smoke configuration is:

```text
backend: ttnn-reference:bf16
batch:   1
warmup:  1
iters:   1
trials:  1
results: results/smoke-tenstorrent
```

Do not begin retained performance testing until top-1, gate, score-error, exact-tie,
padding, and threshold-boundary tests pass.

## 7. Thermal telemetry

The retained benchmark path uses the passive Prometheus endpoint exposed by
`tt-telemetry`. Verify the collector before running a benchmark:

```bash
export TT_TELEMETRY_ENDPOINT=http://127.0.0.1:8080/metrics
curl -fsS "$TT_TELEMETRY_ENDPOINT" | grep -E \
  '^(tt_asic_temperature_celsius|tt_tdp_w|tt_ai_clock_mhz|tt_dispatch_core_efficiency_percent)'
```

The benchmark harness samples Tenstorrent telemetry with:

```bash
--telemetry-provider tenstorrent \
--telemetry-device 0 \
--telemetry-interval-ms 500
```

It records `tt_asic_temperature_celsius`, `tt_tdp_w`, `tt_ai_clock_mhz`, DRAM speed,
dispatch efficiency, and the thermal throttle limit in `rag_telemetry.csv`, then writes
aggregates to `rag_thermal_layer.csv`. The device index selects the sorted `(tray, chip)`
identity from the Prometheus labels. Sampling covers the complete benchmark process.

Use `--telemetry-provider tenstorrent-smi` only as an explicit compatibility fallback.
Do not silently mix `tt-smi` samples with retained `tt-telemetry` results.

## 8. CPU affinity and PCIe NUMA placement

CPU pinning is unnecessary for correctness. For retained measurements, identify the
p150a PCIe function and its NUMA node:

```bash
lspci -D -d 1e52:
cat /sys/bus/pci/devices/0000:BB:DD.F/numa_node
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
```

Start with NUMA-local CPU and memory placement:

```bash
numactl --cpunodebind=<node> --membind=<node> <benchmark-command>
```

Only add `taskset` after comparing default, NUMA-local, and fixed-core-set variance. Give
TT-NN/Metal helper threads several physical cores; do not pin the process to one CPU.
Record the chosen NUMA policy, CPU mask, governor, SMT state, and background-load policy.

## 9. Profiling

Use vendor-native device tools:

- Tracy for the TT-NN/Metal host-device timeline and signposts.
- TT Device Program Profiler for marked device regions and per-RISC/core durations.
- TT-NN operation reports and Visualizer for tensor layouts, conversions, buffers, and
  operation flow.
- NoC profiling/debug dump for bank streaming, multicast, and reduction diagnosis.
- Watcher, DPRINT, asserts, and debug checkpoints only for correctness/debugging.

Use stable regions such as:

```text
rag/request
rag/normalize
rag/prepare
rag/h2d
rag/matmul
rag/d2h_wait
rag/host_reduction
```

The TT-NN adapter exposes these as stable Python functions and optional begin/end
signposts. Capture a short, warmed diagnostic run from the activated TT-Metal Python
environment:

```bash
mkdir -p results/profiles/tenstorrent
python -m tracy -p -r -v tools/ttnn_bench.py \
  bench_data/sweep/entries-3400/bank.bin \
  bench_data/sweep/entries-3400/queries.f32 \
  bench_data/sweep/entries-3400/expected.json \
  --backend ttnn-reference --precision bf16 \
  --batch 32 --warmup 10 --iters 5 \
  --tracy-signposts \
  > results/profiles/tenstorrent/ttnn-reference-bf16-b32.csv
```

`_normalize_stage`, `_prepare_stage`, `_upload_stage`, `_matmul_stage`,
`_download_stage`, and `_reduction_stage` appear as host zones when `-p` is enabled.
The begin/end signposts correlate logical requests with TT-NN operation reports. They
do not synchronize the device. The blocking download remains the only natural request
synchronization point, so the `matmul` host zone may contain enqueue time while
`d2h_wait` may absorb queued device execution.

Profiling runs are diagnostic and must remain separate from release latency results.
Do not pass `--tracy-signposts` to retained release runs.

Tracy may optionally instrument host code on NVIDIA too, but it is not the shared device
profiler: use Tracy/device-profiler tools for Tenstorrent and NVTX/Nsight for NVIDIA.

## 10. Optimized TT-Metalium target

The current development checkpoint is a multicore scoring diagnostic. It discovers
the runtime compute/storage grid and shards contiguous output tiles across active
cores, but it still downloads the complete score tensor for host-side comparison.
Build it on the pinned p150a stack before starting local-winner fusion:

```bash
cd /path/to/tenstorrent2/rag
cmake -S . -B build-tt-metal -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DRAG_ENABLE_TT_METAL=ON \
  -DRAG_TT_METAL_HOME="$TT_METAL_HOME"
cmake --build build-tt-metal --target rag_tt_metal_smoke -j

# v0.80 resolves soc_descriptors relative to TT_METAL_HOME.
RAG_ROOT="$PWD"
cd "$TT_METAL_HOME"
"$RAG_ROOT/build-tt-metal/rag_tt_metal_smoke" \
  "$RAG_ROOT/bench_data/bank.bin" \
  "$RAG_ROOT/bench_data/queries.f32" 0.7
```

Retain the printed runtime grid, active-core count, expected result, and Metalium
result. The run passes only when the ID and gate match the CPU result. Before local
winner fusion, also validate bank sizes `1, 31, 32, 33, 3400`; these cover a partial
tile, exact tile boundaries, and uneven distribution over the runtime grid.

The final backend will:

1. Keep normalized BF16/BFP8 keys resident in p150a GDDR6.
2. Query the available Tensix grid at runtime.
3. Shard contiguous bank ID ranges across participating cores.
4. Stream keys through reader kernels and circular buffers.
5. Compute dot products while maintaining per-core `{score, id}` winners.
6. Perform a deterministic global reduction, choosing the lower ID on an exact tie.
7. Apply `score >= threshold` on device.
8. Return one compact `{accepted, id, score}` result per query.
9. Optionally gather the selected `[101,1024]` visual features on device.

It must not materialize or transfer the complete score matrix on the normal optimized
path.

## 11. Retained benchmark contract

When the TT-NN command adapter and Metalium backend are complete, use:

- Batches: `1,2,4,8,16,32,64,128`.
- Warmup: 10 corpus passes, sufficient to populate program caches.
- Measurement: 100 corpus passes per trial.
- Trials: 10 fresh processes.
- Thermal interval: 500 ms.
- Precision: BF16 first; BFP8 only after a separately approved quality gate.

Keep results separate:

```text
results/tenstorrent-release
results/tenstorrent-device-timing
results/profiles/tenstorrent
```

After the 3,400-entry retained baseline passes, run the complete scaling campaign with:

```bash
python tools/run_bank_sweep.py \
  --rag-bench build-cpu/rag_bench \
  --ttnn-bench tools/ttnn_bench.py \
  --data-root bench_data/sweep \
  --results-root results/tenstorrent-scaling \
  --sizes 10,100,3400,10000 \
  --backends ttnn-reference:bf16 \
  --batches 1,2,4,8,16,32,64,128 \
  --trials 10 --warmup 10 --iters 100 \
  --telemetry-provider tenstorrent \
  --telemetry-device 0 --telemetry-interval-ms 500 \
  --skip-generation
```

Use the same portable bank hash, queries, threshold, batches, warmup policy, and
host-visible timing boundaries as the NVIDIA campaign. Release `steady_clock` results are
authoritative; profiler durations are diagnostic.
