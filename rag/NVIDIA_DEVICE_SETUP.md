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

Stop `tegrastats` after confirming it reports data. Record the JetPack, CUDA, driver,
device, power-mode, and clock configuration with retained results.

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

```bash
python tools/bench_harness.py \
  --rag-bench build-cuda/rag_bench \
  --bank bench_data/bank.bin \
  --queries bench_data/queries.f32 \
  --expected bench_data/expected.json \
  --backends cpu,cuda-reference,cuda-optimized,cuda-optimized:bf16 \
  --batches 1 --warmup 1 --iters 1 --trials 1 \
  --telemetry-provider jetson \
  --results results/smoke-nvidia
```

Use `--telemetry-provider nvidia` instead of `jetson` on systems where `nvidia-smi`
reports temperature, power, clocks, utilization, and thermal slowdown state. Inspect
`rag_telemetry.csv` for sampling errors before starting the retained campaign.

## 5. Authoritative latency and thermal run

```bash
python tools/bench_harness.py \
  --rag-bench build-cuda/rag_bench \
  --bank bench_data/bank.bin \
  --queries bench_data/queries.f32 \
  --expected bench_data/expected.json \
  --backends cpu,cuda-reference,cuda-optimized,cuda-optimized:bf16 \
  --batches 1,2,4,8,16,32,64,128 \
  --warmup 10 --iters 100 --trials 10 \
  --telemetry-provider jetson --telemetry-interval-ms 500 \
  --results results/nvidia-release
```

Thermal sampling runs outside the benchmark child process and does not change its
`steady_clock` timing boundary. It covers the complete child process, including startup,
so current power samples are not converted into energy per query.

## 6. CUDA-event hardware timing

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
  --bank bench_data/bank.bin \
  --queries bench_data/queries.f32 \
  --expected bench_data/expected.json \
  --backends cuda-optimized,cuda-optimized:bf16 \
  --batches 1,2,4,8,16,32,64,128 \
  --warmup 10 --iters 100 --trials 10 \
  --results results/nvidia-cuda-timing
```

This run supplies H2D, conversion, GEMM, reduction, D2H, and total device-stage metrics.
It does not supply published application latency.

## 7. Nsight Systems and NVTX

```bash
cmake -S . -B build-cuda-nvtx -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAG_ENABLE_CUDA=ON \
  -DRAG_ENABLE_NVTX=ON \
  -DRAG_CUDA_ARCHITECTURES=87
cmake --build build-cuda-nvtx
mkdir -p results/profiles/nvidia

nsys profile \
  --trace=cuda,nvtx,cublas,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=results/profiles/nvidia/cuda-optimized-fp32-b32 \
  build-cuda-nvtx/rag_bench \
  bench_data/bank.bin bench_data/queries.f32 bench_data/expected.json \
  --backend cuda-optimized --precision fp32 \
  --batch 32 --warmup 10 --iters 5

nsys stats results/profiles/nvidia/cuda-optimized-fp32-b32.nsys-rep
```

Use Nsight Systems to inspect CPU/CUDA timelines, copies, streams, launch gaps, and NVTX
ranges. Use Nsight Compute only after the timeline identifies a kernel that warrants
counter-level analysis. Neither tool supplies published application percentiles.

## 8. Report and artifacts

```bash
python tools/render_report.py \
  --results results/nvidia-release \
  --report docs/performance_evaluation.md
```

Retain these layers separately:

- `results/nvidia-release`: application latency and thermal telemetry.
- `results/nvidia-cuda-timing`: CUDA-event device stages.
- `results/profiles/nvidia`: diagnostic Nsight reports.

## 9. Real data

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
