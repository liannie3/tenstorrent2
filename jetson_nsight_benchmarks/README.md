# TinyLlama: three Nsight experiments on Jetson AGX Orin

All three tests produce native **.nsys-rep** reports. No benchmark JSON/CSV is
exported. Answers print after capture. CPU and latency numeric measurements are
NVTX text markers inside the report, not automatic numeric plots.

## 1. One-time setup

Extract this folder onto the Jetson, or commit the folder to your GitHub repo
and clone/pull it on the Jetson. Open a terminal in this folder. Activate the
same Python environment that already ran your earlier benchmark:

```bash
source /your/path/to/venv/bin/activate
bash setup_python.sh
```

Replace the venv path with your actual environment. The script checks CUDA
PyTorch before installing Python requirements. It does not install/replace torch.
The requirements preserve your earlier model-library versions and target Python
3.10–3.12. If setup changes a shared environment, use a dedicated project venv
with the compatible Jetson PyTorch build installed.

If starting on a fresh Jetson, install JetPack and the matching NVIDIA PyTorch
build first. Match the JetPack and Python versions; a generic desktop pip torch
wheel is not the Jetson installation procedure. Follow NVIDIA's version-specific
instructions, then run setup_python.sh:
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/

Nsight Systems CLI must also be installed for Jetson/Linux aarch64 and on PATH.
It is not a Python dependency. Check with `nsys --version`. If it is installed
but not on PATH, add the actual directory containing nsys to PATH. Do not install
a desktop x86_64 executable on the Jetson. For CPU profiling diagnostics, run:

```bash
nsys status --environment
```

Use the Nsight version supplied for your JetPack/platform. Permissions and
available channels vary. The package does not change system permissions.

## 2. Get model weights once

If your model is already downloaded, reuse its absolute folder path. Otherwise,
with internet access on the Jetson:

```bash
python download_model.py --destination models/tinyllama
```

This downloads TinyLlama/TinyLlama-1.1B-Chat-v1.0, including safetensors and tokenizer
files, and records the resolved revision. The download needs several GB of disk
space. Subsequent inference uses local files only. Weights are not bundled.

## 3. Check and run ordinary inference

```bash
python check_jetson.py
python -B run.py --model models/tinyllama --text-only --question "What is a pedestrian?" --max-new-tokens 32
```

Normal run.py inference does not launch Nsight. First check confirms a CUDA
operation and NVTX work, but cannot guarantee every Nsight capture channel works.

## 4. Run separate Nsight tests

Start with a small GPU smoke test:

```bash
python -B benchmark_gpu.py --model models/tinyllama --text-only --runs 1 --warmup 1 --max-new-tokens 8 --output results/smoke.nsys-rep
```

Then run the three experiments using the same workload:

```bash
python -B benchmark_latency.py --model models/tinyllama --text-only --runs 3 --output results/latency.nsys-rep
python -B benchmark_cpu.py --model models/tinyllama --text-only --runs 3 --output results/cpu.nsys-rep
python -B benchmark_gpu.py --model models/tinyllama --text-only --runs 3 --output results/gpu.nsys-rep
```

Or run all three sequentially into a new timestamped results directory:

```bash
python -B run_all.py --model models/tinyllama
```

Defaults: same car/pedestrian question, 2 warm-up requests, 3 captured requests,
64 maximum new tokens. Add --question "..." and --max-new-tokens N to change the
workload. EOS may end the answer early. Use new report names for repeated runs;
existing results are preserved. Loading and warm-up are excluded from capture.

If CPU sampling/context-switch capture is unavailable or fails due to permissions,
retry with an explicit reduced capture:

```bash
python -B benchmark_cpu.py --model models/tinyllama --text-only --no-cpu-sampling --output results/cpu_reduced.nsys-rep
```

This retains NVTX process CPU-time annotations and CUDA/OS runtime traces, but
has no native CPU samples or scheduling trace. run_all.py accepts the same flag.
Inspect Nsight warnings even when capture exits successfully: native channels
can be unavailable. The program does not claim sampling succeeded based solely
on the presence of a report.

## 5. What each test measures

| Test | Nsight configuration | Interpretation |
|---|---|---|
| Latency | CUDA + NVTX, no CPU sampling, CUDA synchronization at each stage boundary | Stage wall time including GPU completion, under profiling |
| CPU work | CUDA + NVTX + OS runtime; CPU samples/context switches by default | Sampled CPU activity plus per-stage process CPU time |
| GPU work | CUDA + NVTX, no CPU sampling or per-stage synchronization | Kernels and transfers correlated with stage launches |

Latency fences intentionally change execution timing. These are diagnostic
stage latencies under Nsight, not an unprofiled production baseline. Parent
ranges include child ranges; do not add request, llm_total and their children.
CPU/GPU tests synchronize only at request boundaries (plus the original model's
existing synchronization, such as .item()). Comparing durations across the three
captures is not a controlled estimate of profiler overhead.

Process CPU time sums user/system time across this process's threads, including
runtime threads, and excludes GPU time and other processes. It is approximately
attributed to stages if background threads are active. CPU percent uses one core
as 100%, so multithreaded work may exceed 100%. CPU host wall time may include GPU
waiting; it is not CPU work. Short stages may receive few/no statistical samples.
The CPU test continues running the model on CUDA; it is not CPU-only inference.

## 6. Labels in every report

| Label | Implementation stage |
|---|---|
| prompt_preparation | Validate inputs and prepare system/user text |
| tokenization | Chat formatting and conversion into CPU token IDs |
| input_transfer | Copy IDs to the CUDA device |
| embedding_and_input_setup | Visual checks, embeddings, concatenation, masks, positions |
| prefill | First full-prompt model forward; create KV cache |
| token_selection | Validate scores, argmax and bring chosen token to CPU |
| decode_step | Extend mask/position and run a cached forward |
| output_detokenization | Convert generated IDs to readable text |
| llm_total | Whole LLM answer call |
| request_N | Whole request, including upstream pipeline if enabled |

Prefill happens once per answer. For N generated tokens there are normally N
selections and N-1 decode steps. EOS counts as a generated token. Detokenization
is different from autoregressive decode inference. These labels describe software
stages, not individual transformer blocks. The retained original run.py --benchmark
workflow still has its original per-block hooks if deeper tracing is needed.

Inspect NVTX markers beginning with `metric` for wall_ms (latency test) or
process_cpu_ms / host_wall_ms / cpu_percent_one_core (CPU test). `llm_tokens`
contains input/output counts, TTFT and TPOT. TTFT ends when the first token reaches
the CPU, not when a word displays. TPOT is the average subsequent token interval;
it is None for one token. Both include instrumentation overhead. `experiment`
records model path, prompt, generation settings and framework/device versions.

Open reports in a compatible Nsight Systems desktop GUI on your PC. Expand the
worker's NVTX ranges and CUDA GPU rows. Use CUDA correlation to relate launches
inside a stage to kernels/copies. NVTX ranges are host intervals; GPU work can
extend beyond them. Tokenization normally launches no CUDA work. Overlapping
prior GPU activity does not make tokenization a GPU operation. Kernel-duration
sums can overlap and are not equal to elapsed GPU busy time. No FLOP count,
occupancy, or per-stage utilization percentage is claimed here.
Optional GPU test flag --gpu-metrics-device all requests supported hardware
metrics; start without it because Jetson/tool support and permissions vary.

## 7. Full visual pipeline

The included adapters.py, contracts.py and pipeline.py retain your interfaces.
Replace --text-only in an individual test with:

```bash
--image scene.jpg --alignment alignment.json --adapters team_adapters:build
```

Supply your team's real adapter module, YOLO/vision/projector weights, trained
alignment manifest and RAG assets/dependencies if used. Those were not supplied
and cannot be bundled as functioning implementations. Text-only benchmarking is
complete with this folder plus the installed dependencies and model download.
The pipeline adds yolo, vision_current, rag_query_embedding, rag_search,
vision_reference and llm_answer labels when those paths execute. RAG query
encoding remains one coarse stage. Keep adapters deterministic/reset their state
as appropriate when running repeated requests.

## 8. Files and checks

- benchmark_latency.py, benchmark_cpu.py, benchmark_gpu.py: separate entry points.
- benchmark_stages.py: Nsight parent launcher and captured worker.
- stage_observer.py: shared stage labels and NVTX metric annotations.
- tinyllama_stage.py: original inference with optional stage boundaries.
- run.py: original normal inference and original optional benchmark entry.
- benchmark_nsight.py, nsight_buffer.py: retained dependencies of original run.py.
- pipeline.py, adapters.py, contracts.py: original pipeline interfaces.
- requirements.txt, setup_python.sh, check_jetson.py: environment preparation/check.
- download_model.py: one-time weight download.
- run_all.py: sequential text-only benchmark suite.
- test_nsight.py: hardware-free observer and launcher tests.

```bash
python -B -m unittest test_nsight.py -v
```

Local checks cover parser/launcher configuration, observer policy and mocked
report retention. Real Jetson inference and Nsight capture have not been run in
the build environment. Keep Jetson power mode, clocks, temperature and background
load consistent between experiments and record these with your results.

NVIDIA Nsight documentation: https://docs.nvidia.com/nsight-systems/UserGuide/

## 9. Synthetic subsystem inputs (real TinyLlama)

This exercises pipeline interfaces with fake YOLO detections, zero visual
embeddings and a fixed RAG result. TinyLlama answer generation AND the text RAG
query embedding run on the real model. No alignment checks are disabled: an
explicitly synthetic manifest supplies matching interface metadata, not evidence
of trained visual alignment. Generated answers may be meaningless. Upstream
stage timings measure these stubs, not YOLO, CLIP/projector or database work.

```bash
python prepare_synthetic.py --model models/tinyllama
python -B run.py --model models/tinyllama --image synthetic_inputs/current.ppm --alignment synthetic_inputs/alignment.json --adapters synthetic_adapters:build --question "Describe the test scene." --max-new-tokens 32
python -B benchmark_gpu.py --model models/tinyllama --image synthetic_inputs/current.ppm --alignment synthetic_inputs/alignment.json --adapters synthetic_adapters:build --question "Describe the test scene." --max-new-tokens 32 --output results/synthetic_gpu.nsys-rep
```

Replace benchmark_gpu.py with benchmark_latency.py or benchmark_cpu.py and choose
new output names for the other tests. Default fixed RAG score 0.9 exercises the
accepted-reference branch. Prefix a command with `SYNTHETIC_RAG_SCORE=0.2` to test
rejection. Synthetic tensors use the contract's real sequence sizes (576 current,
101 retrieved), but their values do not represent images. Black 1x1 PPM fixtures
are valid files; the synthetic vision adapter intentionally ignores their pixels.
