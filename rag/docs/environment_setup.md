# Development Environment Setup

This project benchmarks conditional RAG retrieval across CPU, NVIDIA CUDA, and
Tenstorrent Blackhole. Use the CPU implementation as the correctness reference before
measuring either accelerator backend.

## Supported development environments

| Environment | Purpose |
| --- | --- |
| Linux host with a Blackhole p150a | Primary Tenstorrent development and benchmarking |
| NVIDIA Jetson with JetPack | CUDA reference development and edge measurements |
| Linux, Windows, or macOS | CPU-only development and bank-format tests |

A Python virtual environment isolates Python packages only. Device drivers, firmware,
the C++ compiler, CMake, CUDA, and TT-Metalium remain system dependencies.
The RAG subsystem is pinned to Python **3.10.12** in `.python-version`.

## 1. Clone the project

```bash
git clone <repository-url> tenstorrent2
cd tenstorrent2/rag
```

Replace `<repository-url>` with the URL for this repository.

## 2. Create the project Python environment

The existing bank builder uses only the Python standard library. A virtual environment
is still useful for tests and future TT-NN integration.

### Ubuntu

```bash
sudo apt update
sudo apt install -y python3.10 python3-pip python3.10-venv build-essential cmake ninja-build

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python --version
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python --version
```

Reactivate the environment in each new terminal:

```bash
source .venv/bin/activate
```

The version check must report Python 3.10.12 before building fixtures or running tests.

## 3. Verify the CPU reference implementation

Configure, build, and test the portable implementation before installing an accelerator
backend:

```bash
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAG_ENABLE_CUDA=OFF

cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

On a multi-configuration generator such as Visual Studio, add `--config Release` to the
build and test commands.

## 4. Set up the Tenstorrent host

The Blackhole p150a is a PCIe accelerator. Install and validate it on a supported Linux
host before installing TT-NN. Do not assume a Jetson can act as the p150a host; confirm
the required PCIe slot, architecture, operating system, driver, and power configuration
with Tenstorrent for the exact machines in the benchmark.

Follow the current official installation guide rather than copying a driver or firmware
version into this repository:

- [TT-NN documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/)
- [TT-Metalium documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/)

Tenstorrent provides wheel, container, source, and Conda installation paths. Use a
pinned release rather than a floating nightly build for reproducible benchmark results.

### Recommended starting point: TT-NN wheel

After the Tenstorrent driver and hardware are working, activate the project environment
and install the release selected by the team:

```bash
source .venv/bin/activate
python -m pip install ttnn
```

Record the resolved version:

```bash
python -c "import ttnn; print(ttnn.__version__)"
python -m pip freeze > environment-lock.txt
```

If the project needs custom TT-Metalium kernels, use Tenstorrent's source installation
instructions instead of mixing a source build with an unrelated wheel. Record the exact
TT-Metal commit hash in every benchmark report.

### Validate the Tenstorrent installation

Run the hardware and example checks supplied by the selected TT-Metal release. Then
verify that Python can import TT-NN:

```bash
python -c "import ttnn; print('TT-NN import succeeded')"
```

The precise device-validation command can change between TT-Metal releases. Use the
validation steps shipped with the pinned release and save their output with the benchmark
artifacts.

## 5. Set up an NVIDIA Jetson reference system

Use the Jetson only for the NVIDIA CUDA comparison unless the final hardware design
explicitly connects it to the Tenstorrent accelerator.

Install the JetPack release appropriate for the Jetson model. JetPack supplies the
compatible NVIDIA driver and CUDA toolkit:

- [NVIDIA JetPack installation](https://docs.nvidia.com/jetson/jetpack/install-setup/index.html)

Confirm that the CUDA compiler is visible:

```bash
nvcc --version
```

Create the Python environment using the Ubuntu commands in section 2, then build the
CUDA backend:

```bash
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAG_ENABLE_CUDA=ON

cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

Do not install an arbitrary desktop CUDA toolkit over JetPack. The CUDA compiler and
runtime must remain compatible with the Jetson driver.

## 6. Prepare test data

Each JSONL memory entry must contain a description, a 2,048-value TinyLlama text key,
and the relative path to a `101 x 1024` little-endian float32 CLIP feature file:

```json
{"description":"A pedestrian is crossing.","text_key":[0.1,0.2],"visual_f32":"features/0001.bin"}
```

The abbreviated `text_key` above illustrates the schema only. Real entries must contain
exactly 2,048 finite values.

Build the bank:

```bash
python tools/build_bank.py entries.jsonl memory.bank \
  --metadata bank_metadata.json
```

The builder creates `memory.bank` and `memory.bank.json`. Preserve both files so model,
preprocessing, and source-data versions can be audited.

Create or obtain a runtime query containing exactly 2,048 little-endian float32 values,
then run the CPU reference:

```bash
./build-cpu/rag_query memory.bank query.f32 0.7
```

Run the CUDA implementation on the Jetson:

```bash
./build-cuda/rag_query memory.bank query.f32 0.7 --cuda
```

## 7. Tenstorrent development progression

Implement the Tenstorrent backend in two stages:

1. Build a TT-NN reference using a resident, transposed key matrix, `ttnn.matmul`, and
   top-1 reduction.
2. After validating numerical behavior, implement a fused TT-Metalium operation that
   performs dot products, top-1 reduction, and thresholding without returning every score
   to the host.

Keep these representations separate:

- `memory.bank`: portable float32 bank and payloads used by CPU and CUDA.
- Tenstorrent packed keys: padded, tiled BF16 or BFP8 keys used for performance tests.
- Manifest: source-bank hash, TT-Metal version, data type, padded shape, and packing
  version.

Never compare reduced-precision output only by cosine-score error. Also measure top-1 ID
agreement and acceptance/rejection agreement around the configured threshold.

## 8. Reproducible benchmark checklist

Organize benchmark output using the Application Layer, Hardware Layer, and Stage Layer
defined in [performance_evaluation.md](performance_evaluation.md). Each recorded metric
must identify its layer, scenario, timing boundary, and whether it is a primary comparison
metric, quality gate, deployment metric, or diagnostic metric.

Record the following with every result:

- Git commit for this repository.
- TT-Metal release or commit and installed firmware/driver versions.
- JetPack and CUDA versions for the NVIDIA result.
- CPU model, accelerator model, system memory, and operating system.
- Bank entry count, embedding dimension, data type, and bank SHA-256 hash.
- Query batch size, warmup count, measured iteration count, and threshold.
- Cold-start time separately from steady-state latency.
- P50, P95, and P99 latency, throughput, and top-1 correctness.
- Whether host-device transfers are included in each reported number.

Use the same bank, queries, acceptance threshold, warmup policy, and timing boundaries on
both accelerators. Compare optimized Tenstorrent code against both the existing CUDA
reference and an optimized CUDA implementation with a persistent bank and device-side
top-1 reduction.

## 9. Common problems

### CMake cannot find CUDA

Confirm that JetPack is installed and that `nvcc --version` succeeds in the same terminal.
Delete only the dedicated `build-cuda` directory after changing toolchains, then configure
it again.

### TT-NN imports but no device is available

Re-run the device checks from the pinned TT-Metal release. Confirm that the driver,
firmware, userspace package, and hardware architecture all belong to the same supported
release combination.

### CPU and accelerator results disagree

Check query normalization, key ordering, padded-entry masking, reduced precision, and
equal-score tie handling. The reference implementation selects the lower entry ID when
scores are equal and accepts a match when `score >= threshold`.
