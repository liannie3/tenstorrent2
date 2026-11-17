# NVIDIA RAG Device Setup

This guide configures an NVIDIA host—especially Jetson AGX Orin—for the CPU and CUDA
retrieval benchmarks. Compiled binaries are not stored in Git and must be rebuilt on
each machine.

## 1. Prerequisites

Install Git, CMake 3.21 or newer, Ninja, Python 3.10 or newer, and a C++17 compiler
supported by the installed CUDA toolkit. On Jetson, use the CUDA toolkit and driver
provided by the matching JetPack release; do not replace them with generic desktop
packages.

Verify the platform:

```bash
nvcc --version
nvidia-smi || true
tegrastats --interval 1000
```

Stop `tegrastats` after confirming it reports data. List the available Jetson power
modes and record the active mode before changing it:

```bash
sudo nvpmodel -q --verbose
sudo nvpmodel -p --verbose
```

Choose the approved project power mode, then hold it constant for every retained run.
If the experiment uses maximum clocks, apply and verify them before both the NVIDIA and
Tenstorrent comparison campaigns:

```bash
sudo nvpmodel -m MODE_ID
sudo jetson_clocks
sudo jetson_clocks --show
```

Do not silently select a power mode: record `MODE_ID`, whether `jetson_clocks` is active,
and the CPU governor in the benchmark manifest using the harness options shown below.

## 2. Clone and prepare Python

```bash
git clone https://github.com/liannie3/tenstorrent2.git
cd tenstorrent2
git switch al/rag-jetson
cd rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

The benchmark orchestration and reporting tools use the Python standard library.

## 3. Build the release benchmark

For Jetson AGX Orin, CUDA architecture 87:

```bash
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAG_ENABLE_CUDA=ON \
  -DRAG_CUDA_ARCHITECTURES=87
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

For a different NVIDIA GPU, replace `87` with its CUDA compute capability. Keep the
release build free of CUDA-event timing and NVTX; it supplies authoritative application
latency and throughput.

Available backends are:

- `cpu`
- `cuda-reference`
- `cuda-optimized`
- `cuda-optimized:bf16`

## 4. Smoke test

Generate the small functional fixture explicitly. The default generator now creates the
3,400-entry baseline and is unnecessarily large for a smoke test:

```bash
python tools/generate_corpus.py \
  --output bench_data/smoke-8 --entries 8
```

```bash
python tools/bench_harness.py \
  --rag-bench build-cuda/rag_bench \
  --bank bench_data/smoke-8/bank.bin \
  --queries bench_data/smoke-8/queries.f32 \
  --expected bench_data/smoke-8/expected.json \
  --backends cpu,cuda-reference,cuda-optimized,cuda-optimized:bf16 \
  --batches 1 --warmup 1 --iters 1 --trials 1 \
  --telemetry-provider jetson \
  --telemetry-power-rail VDD_GPU_SOC \
  --run-kind smoke
```

Use `--telemetry-provider nvidia` instead of `jetson` on systems where `nvidia-smi`
reports temperature, power, clocks, utilization, and thermal slowdown state. Inspect
`rag_telemetry.csv` for sampling errors before starting the retained campaign.
On Jetson, the same run also writes every power rail present in each `tegrastats`
sample to `rag_power_rails.csv`; no per-rail benchmark reruns are required.
Confirm the selected rail exists on the device. Use `VIN_SYS_5V0` when the study requires
the broadest available module-level boundary; use `VDD_GPU_SOC` only for the GPU/SOC rail.
Never compare energy values collected from different rail boundaries.

## 5. Generate the baseline and scaling banks

The standard sweep is 10, 100, 3,400, and 10,000 entries. Generation writes about
5.4 GiB of bank payloads in total, plus temporary JSONL data while each bank is packed.
Verify free disk space first:

```bash
df -h .
python tools/generate_corpus.py \
  --output bench_data/sweep \
  --sweep 10,100,3400,10000
```

Each `entries-N` directory contains a bank, queries, expected results, and hashes. The
3,400-entry bank is the paper-comparison baseline. The 8-entry bank remains a smoke
fixture and must not be reported as platform performance.

## 6. Authoritative release and telemetry campaign

Set a single UTC run date so release, timing, profiles, and reports cannot split across
two dated directories if the campaign crosses midnight:

```bash
export RUN_DATE=$(date -u +%F)
```

Run the 3,400-entry baseline first. Use a measured phase long enough to place at least two
telemetry samples inside every trial; otherwise J/query is deliberately left empty.

```bash
python tools/bench_harness.py \
  --rag-bench build-cuda/rag_bench \
  --bank bench_data/sweep/entries-3400/bank.bin \
  --queries bench_data/sweep/entries-3400/queries.f32 \
  --expected bench_data/sweep/entries-3400/expected.json \
  --backends cpu,cuda-reference,cuda-optimized,cuda-optimized:bf16 \
  --batches 1,4,16,32,128 \
  --warmup 10 --iters 100 --trials 10 \
  --telemetry-provider jetson --telemetry-interval-ms 500 \
  --telemetry-power-rail VDD_GPU_SOC \
  --power-mode MODE_ID \
  --clock-policy jetson_clocks \
  --cpu-governor GOVERNOR \
  --run-date "$RUN_DATE" \
  --run-kind release
```

Thermal sampling runs outside the benchmark child process and does not change its
`steady_clock` timing boundary. The raw file retains complete-process telemetry, while
energy is trapezoid-integrated only over samples assigned to the final measured phase.
Inspect `rag_telemetry.csv`, `rag_power_rails.csv`, and `rag_thermal_layer.csv`. Treat
J/query as unavailable when there are fewer than two measured samples, sampling errors,
or a maximum sample gap too large for the workload duration.

Validate the retained baseline before reporting it:

```bash
python tools/validate_campaign.py "results/$RUN_DATE/release"
```

The validator requires at least 3,400 entries, ten trials, batches 1/4/16/32/128, passing
quality gates, and throughput CV no greater than 5%.

After the baseline passes, run the remaining bank sizes with `tools/run_bank_sweep.py`,
or invoke the harness per existing `entries-N` directory when the banks were generated
in advance. Keep each size in a separate result directory.

```bash
python tools/run_bank_sweep.py \
  --rag-bench build-cuda/rag_bench \
  --data-root bench_data/sweep \
  --results-root "results/$RUN_DATE/scaling" \
  --sizes 10,100,3400,10000 \
  --backends cpu,cuda-optimized,cuda-optimized:bf16 \
  --batches 1,4,16,32,128 \
  --trials 10 --warmup 10 --iters 100 \
  --telemetry-provider jetson \
  --telemetry-power-rail VDD_GPU_SOC \
  --power-mode MODE_ID \
  --clock-policy jetson_clocks \
  --cpu-governor GOVERNOR \
  --skip-generation
```

## 7. CUDA-event hardware timing

Use a separate build so event overhead cannot affect release results:

```bash
cmake -S . -B build-cuda-timing -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAG_ENABLE_CUDA=ON \
  -DRAG_ENABLE_CUDA_TIMING=ON \
  -DRAG_CUDA_ARCHITECTURES=87
cmake --build build-cuda-timing
ctest --test-dir build-cuda-timing --output-on-failure

python tools/bench_harness.py \
  --rag-bench build-cuda-timing/rag_bench \
  --bank bench_data/sweep/entries-3400/bank.bin \
  --queries bench_data/sweep/entries-3400/queries.f32 \
  --expected bench_data/sweep/entries-3400/expected.json \
  --backends cuda-optimized,cuda-optimized:bf16 \
  --batches 1,4,16,32,128 \
  --warmup 10 --iters 100 --trials 10 \
  --power-mode MODE_ID \
  --clock-policy jetson_clocks \
  --cpu-governor GOVERNOR \
  --run-date "$RUN_DATE" \
  --run-kind cuda-timing
```

This run supplies H2D, conversion, GEMM, reduction, D2H, and total device-stage
P50/P95/P99 metrics, plus persistent, peak, and workspace device-memory accounting. It
does not supply published application latency.

## 8. Nsight Systems, Nsight Compute, and NVTX

```bash
cmake -S . -B build-cuda-nvtx -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAG_ENABLE_CUDA=ON \
  -DRAG_ENABLE_NVTX=ON \
  -DRAG_CUDA_ARCHITECTURES=87
cmake --build build-cuda-nvtx
mkdir -p "results/$RUN_DATE/profiles"

nsys profile \
  --trace=cuda,nvtx,cublas,osrt \
  --sample=none \
  --force-overwrite=true \
  --output="results/$RUN_DATE/profiles/cuda-optimized-fp32-b32" \
  build-cuda-nvtx/rag_bench \
  bench_data/sweep/entries-3400/bank.bin \
  bench_data/sweep/entries-3400/queries.f32 \
  bench_data/sweep/entries-3400/expected.json \
  --backend cuda-optimized --precision fp32 \
  --batch 32 --warmup 10 --iters 5

nsys stats "results/$RUN_DATE/profiles/cuda-optimized-fp32-b32.nsys-rep" \
  > "results/$RUN_DATE/profiles/nsys-stats.txt"
```

Use Nsight Systems to inspect CPU/CUDA timelines, copies, streams, launch gaps, and NVTX
ranges. Use Nsight Compute only after the timeline identifies a kernel that warrants
counter-level analysis. Neither tool supplies published application percentiles.

For one representative FP32 and BF16 configuration, collect an Nsight Compute report
containing the SpeedOfLight and MemoryWorkloadAnalysis sections. Keep the capture short:

```bash
ncu --set full --target-processes all \
  --export "results/$RUN_DATE/profiles/cuda-optimized-fp32-b32" \
  build-cuda/rag_bench \
  bench_data/sweep/entries-3400/bank.bin \
  bench_data/sweep/entries-3400/queries.f32 \
  bench_data/sweep/entries-3400/expected.json \
  --backend cuda-optimized --precision fp32 \
  --batch 32 --warmup 2 --iters 1
```

Nsight Compute is diagnostic. Do not use its request duration as release latency.

## 9. Report and artifacts

```bash
python tools/render_report.py \
  --run-dir "results/$RUN_DATE"
python tools/plot_results.py "results/$RUN_DATE"
```

Retain these layers separately inside the same dated bundle:

- `results/YYYY-MM-DD/release`: application latency and thermal telemetry.
- `results/YYYY-MM-DD/cuda-timing`: CUDA-event device stages.
- `results/YYYY-MM-DD/profiles`: diagnostic Nsight reports and saved `nsys stats` output.
- `results/YYYY-MM-DD/report`: combined report and presentation charts.

Also retain `rag_run_manifest.json`, all raw trial CSVs, `rag_power_rails.csv`, and the
output of `tools/validate_campaign.py` with the campaign archive.

## 10. Real data

Generate the deterministic fixture only when intentionally replacing it:

```bash
python tools/generate_corpus.py
```

Build a real bank with:

```bash
python tools/build_bank.py entries.jsonl memory.bank \
  --metadata bank_metadata.json
```

Queries contain exactly 2,048 little-endian float32 values per embedding. Use the same
bank hash, query corpus, threshold, batches, warmup policy, and timing boundary in the
Tenstorrent campaign.
