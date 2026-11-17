# Development Environment

## Target platform

- NVIDIA Jetson supported by JetPack **6.2.1+b38**.
- 64-bit ARM Linux (`aarch64`).
- CUDA toolkit and NVIDIA drivers supplied by the JetPack installation.
- CMake 3.21 or newer.
- A C++17-compatible host compiler supported by the installed CUDA toolkit.

Do not replace JetPack's CUDA toolkit or NVIDIA drivers with generic desktop packages.
The CUDA compiler, runtime, and driver must remain compatible with the installed JetPack
release.

## Python

The project development environment uses Python **3.10.12**. This is a project constraint,
not a claim that JetPack itself requires that Python version.

Confirm the interpreter selected on the development system:

```bash
python3 --version
which python3
python3 -c "import sys; print(sys.executable); print(sys.version)"
```

Create and activate the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

The final command must report:

```text
Python 3.10.12
```

The current bank-building and Python test scripts use only the Python standard library,
so the project does not currently require a `requirements.txt` file.

## Verify the toolchain

```bash
python --version
cmake --version
nvcc --version
```

Record the complete output of these commands with benchmark results.

## CPU reference build

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAG_ENABLE_CUDA=OFF
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

## Jetson CUDA build

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAG_ENABLE_CUDA=ON
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

## CUDA timing build

CUDA-event timing is optional. It records H2D, BF16 conversion, GEMM, top-1 reduction,
D2H, and total device-pipeline durations in the benchmark CSV.

```bash
cmake -S . -B build-cuda-timing \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAG_ENABLE_CUDA=ON \
  -DRAG_ENABLE_CUDA_TIMING=ON
cmake --build build-cuda-timing
ctest --test-dir build-cuda-timing --output-on-failure
```

Use the ordinary CUDA build, with event timing disabled, for authoritative application
latency and throughput. Use the timing build for hardware-layer results until an A/B study
shows that event recording stays within the project's accepted overhead budget.

## NVTX and Nsight Systems

Build NVTX annotations separately from the authoritative latency build:

```bash
cmake -S . -B build-cuda-nvtx \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAG_ENABLE_CUDA=ON \
  -DRAG_ENABLE_NVTX=ON
cmake --build build-cuda-nvtx
```

Capture a short representative run after generating or selecting the benchmark corpus:

```bash
mkdir -p results/profiles
nsys profile \
  --trace=cuda,nvtx,cublas,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=results/profiles/cuda-optimized-fp32-b32 \
  build-cuda-nvtx/rag_bench \
  bench_data/bank.bin bench_data/queries.f32 bench_data/expected.json \
  --backend cuda-optimized --precision fp32 \
  --batch 32 --warmup 10 --iters 5
```

Open the resulting `.nsys-rep` in Nsight Systems or export summary statistics with the
installed `nsys stats` command. Confirm command-line options against the Nsight Systems
version installed on the Jetson.

Nsight captures are diagnostic artifacts. Do not use their host durations as the source
of application P50/P95/P99 results. Record the Nsight, CUDA, driver, device, power-mode,
and clock-setting versions alongside every retained capture.

## Thermal telemetry

The benchmark harness can sample accelerator temperature, power, clocks, utilization,
and reported throttling without changing the application timing boundary:

```bash
# Jetson through tegrastats
python tools/bench_harness.py ... --telemetry-provider jetson

# Select a different tegrastats power rail (use the exact name printed by tegrastats)
python tools/bench_harness.py ... --telemetry-provider jetson --telemetry-power-rail VDD_CPU_CV

Every Jetson sample captures all power rails reported on that same `tegrastats` line.
`rag_power_rails.csv` contains one time-aligned row per rail. The selected rail controls
only the backward-compatible `power_w` column and aggregate thermal report; it does not
limit collection.

# NVIDIA systems with nvidia-smi telemetry support
python tools/bench_harness.py ... --telemetry-provider nvidia

# Tenstorrent through structured tt-smi snapshots
python tools/bench_harness.py ... --telemetry-provider tenstorrent
```

Use `--telemetry-device N` and `--telemetry-interval-ms N` to select the device and
sampling interval. The interval must be at least 100 ms and defaults to 500 ms. Raw
samples are written to `rag_telemetry.csv`; aggregated temperature, power, clock, and
throttling values are written to `rag_thermal_layer.csv` and rendered in the report.

Sampling covers the complete benchmark child process, including startup. Therefore these
power samples are not used to calculate energy per query. A sampling failure is recorded
in the raw CSV and does not become a zero or terminate the benchmark.

Tenstorrent telemetry tooling can vary by installed stack, and some versions may contend
with an exclusively opened device. Validate `tt-smi -s` concurrently with a short smoke
run before the retained campaign. If it cannot sample concurrently, use the passive
sysfs/tt-toplike backend supplied by that host and adapt it behind the same CSV schema.

Tracy is optional for comparing host-thread behavior, but device diagnostics remain
vendor-native: Tracy and the TT device profiler for Tenstorrent; NVTX, Nsight Systems,
and Nsight Compute for NVIDIA. Neither profiler supplies authoritative application
latency.
