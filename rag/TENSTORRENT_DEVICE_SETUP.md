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
tt-smi -s
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

The repository currently contains the resident-bank implementation in
`rag/tools/ttnn_backend.py` and host-only contract tests. The `rag_bench`-compatible
TT-NN command adapter is the next implementation milestone. Until it lands, validate the
module and installed API directly:

```bash
python tests/test_ttnn_backend.py
python -c "from pathlib import Path; from tools.ttnn_backend import TtnnReferenceSearcher; s=TtnnReferenceSearcher(Path('bench_data/bank.bin')); print(s.entries, s.dimension, s.padded_entries); s.close()"
```

The reference backend keeps a transposed BF16 bank resident, performs TT-NN matmul, and
currently returns the score matrix for deterministic host-side top-1. It is a correctness
baseline, not the final performance design.

Once the command adapter is available, its required smoke configuration is:

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

First verify that structured snapshots work:

```bash
tt-smi -s | python -m json.tool
```

The benchmark harness samples Tenstorrent telemetry with:

```bash
--telemetry-provider tenstorrent \
--telemetry-device 0 \
--telemetry-interval-ms 500
```

It records raw samples in `rag_telemetry.csv` and aggregated temperature, power, clock,
and throttling results in `rag_thermal_layer.csv`. Sampling covers the entire benchmark
child process and is not currently used to calculate energy per query.

Some TT-SMI/driver versions may contend with an exclusively opened device. Validate
concurrent `tt-smi -s` sampling during a short TT-NN smoke test before the retained run.
If samples fail, preserve the errors and use a passive sysfs or supported tt-toplike
backend behind the same CSV schema rather than reporting zeros.

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
rag/query_prepare
rag/h2d
rag/dot_product
rag/local_top1
rag/global_reduction
rag/d2h
rag/result_construct
```

Exact profiler commands depend on the installed TT-Metal build and are finalized after
the host inventory. Profiling runs are diagnostic and must remain separate from release
latency results.

Tracy may optionally instrument host code on NVIDIA too, but it is not the shared device
profiler: use Tracy/device-profiler tools for Tenstorrent and NVTX/Nsight for NVIDIA.

## 10. Optimized TT-Metalium target

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

Use the same portable bank hash, queries, threshold, batches, warmup policy, and
host-visible timing boundaries as the NVIDIA campaign. Release `steady_clock` results are
authoritative; profiler durations are diagnostic.
