# TinyLlama on Jetson AGX Orin — Nsight + Temperature + Energy Efficiency

**New:** read [EFFICIENCY.md](EFFICIENCY.md) for energy per request, joules/token, tokens/joule, power and clock summaries, and sample coverage rules. Same `--tegrastats` flag, no new libraries.

This folder packages the **user's latest TinyLlama implementation**, preserving its explanatory comments and section dividers, and adds optional Jetson temperature sampling through `tegrastats`.

**What is included:** text-only TinyLlama inference; pipeline interfaces for real teammate-supplied YOLO/CLIP/RAG components; ordinary Nsight profiling; separate latency/CPU/GPU experiments; a sequential three-experiment suite; and request-scoped hardware telemetry.

**What is not included:** model weights; an actual trained visual adapter/projector; Jetson CUDA/PyTorch or `nsys` installations. Those must already exist or be supplied separately. No Jetson hardware run was possible here; the local tests are synthetic and do not establish hardware performance.

## 1. Folder map

```text
jetson_tinyllama_efficiency/
├── run.py                       # Original single-run + optional Nsight path
├── tinyllama_stage.py           # Your latest commented TinyLlama inference code
├── pipeline.py                  # Your commented YOLO -> Vision -> RAG -> TinyLlama path
├── adapters.py                  # Team subsystem adapters (real functions required)
├── contracts.py                 # Dataclasses / interface contracts
├── nsight_buffer.py             # Your NVTX hooks + CPU/RAM + NEW temperature wiring
├── tegrastats_monitor.py        # NEW: temperature / clocks / GPU / power parser
├── benchmark_nsight.py          # Parent/child Nsight launcher for run.py
├── stage_observer.py            # NEW: missing link required by benchmark_stages.py
├── benchmark_stages.py          # Original three individual benchmark experiments
├── benchmark_latency.py         # Original latency launcher
├── benchmark_cpu.py             # NEW: missing CPU launcher
├── benchmark_gpu.py             # NEW: missing GPU launcher
├── run_all.py                   # Original sequential suite, now forwards telemetry
├── check_jetson.py              # Jetson environment check
├── download_model.py            # Optional one-time Hugging Face model download
├── requirements.txt             # User's pinned library baseline, excludes torch
├── test_tegrastats_monitor.py   # CPU-only telemetry parser/subprocess tests
├── test_benchmark_wiring.py     # NEW: CPU-only integration/wiring tests
├── CHANGES.md                   # File-by-file summary of updates
├── models/                      # Put your local TinyLlama checkpoint here
└── results/                     # Nsight report output folder
```

**The newest user-provided section headers are retained** in `tinyllama_stage.py` and `pipeline.py`, including the dashed/hash separators before prompt preparation, tokenization, embedding setup, prefill, token selection, decode, and detokenization. The neural-network inference algorithm itself was not modified.

## 2. On your Jetson: unpack and check the environment

```bash
unzip jetson_tinyllama_efficiency.zip
cd jetson_tinyllama_efficiency
python3 --version
python3 check_jetson.py
which nsys
nsys --version
which tegrastats
```

`check_jetson.py` should report a working CUDA-enabled PyTorch build and a successful FP16 GPU calculation. You must use a JetPack-compatible PyTorch build; **do not replace working NVIDIA PyTorch with an arbitrary `pip install torch`**.

In an existing virtual environment that already has CUDA-enabled PyTorch:

```bash
python3 -m pip install -r requirements.txt
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If you need a new virtual environment, use your JetPack-compatible PyTorch installation procedure first. The pinned Transformers dependencies in this folder are an older project baseline, not a guarantee of compatibility with every current JetPack release.

For an initial sensor-only check, run the NVIDIA command (press Ctrl+C to stop):

```bash
tegrastats --interval 100
```

Jetson thermal zone names vary: `GPU@...C`, `CPU@...C`, `SOC0@...C`, etc. Only fields reported by your board can be recorded.

## 3. Prepare the model once

If your model weights are **already on the Jetson**, keep them where they are and pass their real location with `--model`.

If you still need to download TinyLlama and have network access:

```bash
python3 download_model.py --destination models/tinyllama-chat
```

The model directory must contain the saved configuration, tokenizer and weights. Downloads and initialization are **excluded** from the benchmark timing.

## 4. Test ordinary TinyLlama inference first

```bash
python3 -B run.py \
  --model models/tinyllama-chat \
  --text-only \
  --question "Describe what a pedestrian is." \
  --max-new-tokens 32
```

**If this fails, fix the model/environment problem before starting Nsight.**

## 5. Single Nsight report with temperature

```bash
mkdir -p results
python3 -B run.py \
  --model models/tinyllama-chat \
  --text-only \
  --question "Describe what a pedestrian is." \
  --max-new-tokens 128 \
  --benchmark results/nsight_temperature_01.nsys-rep \
  --warmup 2 --runs 3 \
  --interval-ms 100 \
  --tegrastats \
  --tegrastats-interval-ms 100
```

The optional `--tegrastats` flag **enables** temperature, GPU/EMC clock and activity, memory, and power-rail telemetry when available. Without it, your original Nsight path remains available.

Reports must use a **new filename** each time; this code deliberately refuses to overwrite an existing `.nsys-rep` report.

The report contains relevant named NVTX events:

- `request_1`, `request_2`, ... and `pipeline_latency_ms=...`
- `application input_tokens=... latency_ms=... ttft_ms=... tpot_ms=...`
- `prompt_preparation`, `tokenization`, `input_transfer`, `embedding_and_input_setup`, `prefill`, `token_selection`, `decode_step`, `output_detokenization`
- Individual transformer `block_00`, `block_01`, ... and forward-hook `prefill`/`decode` labels
- `resources ...` for CPU/RAM sampling
- `tegrastats ... gpu_temp_c=... cpu_temp_c=... soc0_temp_c=...` if those sensors exist
- `tegrastats_summary ... gpu_temp_mean_c=... gpu_temp_min_c=... gpu_temp_max_c=... gpu_temp_samples=N ...`
- `pytorch_gpu_memory ...` for PyTorch CUDA allocator values

**Reading the report:** Open the `.nsys-rep` in Nsight Systems, expand NVTX tracks, and search for `tegrastats_summary` and `tegrastats` markers. The telemetry monitor writes NVTX annotations only; this code does not automatically generate CSV or a temperature graph.

## 6. Run the latency, CPU, or GPU experiment individually

Each script loads the **same TinyLlama model**. The main differences are the Nsight trace/sampling settings and the report filenames, not three different inference algorithms.

```bash
# Latency experiment
python3 -B benchmark_latency.py \
  --model models/tinyllama-chat --text-only \
  --max-new-tokens 128 --warmup 2 --runs 3 \
  --output results/latency_temperature_01.nsys-rep \
  --tegrastats --tegrastats-interval-ms 100

# CPU experiment (native Nsight CPU sampling may need permissions)
python3 -B benchmark_cpu.py \
  --model models/tinyllama-chat --text-only \
  --max-new-tokens 128 --warmup 2 --runs 3 \
  --output results/cpu_temperature_01.nsys-rep \
  --tegrastats --tegrastats-interval-ms 100

# GPU experiment
python3 -B benchmark_gpu.py \
  --model models/tinyllama-chat --text-only \
  --max-new-tokens 128 --warmup 2 --runs 3 \
  --output results/gpu_temperature_01.nsys-rep \
  --tegrastats --tegrastats-interval-ms 100
```

If native CPU stack sampling fails due to permissions or Nsight support, retry the CPU command with a new filename and `--no-cpu-sampling`. Linux `/proc` CPU/RAM annotations will still be collected.

The **GPU experiment** now also enables Nsight CUDA memory-allocation tracing and records a PyTorch CUDA memory summary. Extra GPU performance counters remain optional and are hardware-dependent (`--gpu-metrics-device 0` only when supported).

## 7. Run all three experiments sequentially

```bash
python3 -B run_all.py \
  --model models/tinyllama-chat \
  --question "Describe what a pedestrian is." \
  --max-new-tokens 128 --warmup 2 --runs 3 \
  --tegrastats --tegrastats-interval-ms 100 \
  --output-dir results/suite_temperature_01
```

`run_all.py` produces three separate reports: `latency.nsys-rep`, `cpu.nsys-rep`, `gpu.nsys-rep`. They are measured **sequentially**, not simultaneously. `run_all.py` follows your uploaded design and is **text-only**. Use `run.py` or the individual `benchmark_*.py` scripts with your teammates' actual adapter factory for multimodal testing.

## 8. How temperature is attached

```text
run.py / benchmark_stages.py
    |  receives --tegrastats
    v
NsightBuffer / StageObserver
    |  pipeline_scope(): begin the measured request
    +--> Linux CPU/RAM monitor thread
    +--> TegrastatsMonitor.start() -> tegrastats --interval 100
    |                                    |-- live sensor readings
    |                                    +-- NVTX: tegrastats ...
    v
TinyLlamaStage.answer() -> _answer_impl()
    |-- existing NVTX stage spans
    |-- forward hooks: prefill / decode / blocks
    |-- token timestamps: TTFT / TPOT / throughput
    v
pipeline_scope(): synchronize and finish
    +--> TegrastatsMonitor.stop()
    |    +-- NVTX: tegrastats_summary (mean/min/max temperature)
    +--> stop CPU/RAM thread
```

Telemetry starts before the timed request and stops after it. Samples are collected independently, so they describe the **whole Jetson** rather than power or temperature attributable exclusively to TinyLlama. The NVTX timestamp marks when a line is received and parsed, not necessarily the exact hardware sensor capture time.

**Caution:** Short requests may finish before the first thermal sample; then the event reports `status=no_samples`. Increase the output-token limit or the number/length of measured workloads instead of inventing temperature values. A 100 ms interval is an intended sampling cadence, not a guarantee.

**Caution:** The uploaded `tinyllama_stage.py` has an explicit `prefill` span and the forward hook also labels the first model pass `prefill`. These **nested labels do not mean two prefill passes**. Similarly, the `decode_step` span includes input setup while the hook's `decode` label surrounds the model forward.

## 9. Run local integration tests

These tests use synthetic telemetry or fake Nsight subprocesses, so they can run on another Linux machine without the Jetson.

```bash
python3 -m unittest discover -v
python3 -m compileall -q .
```

They validate parsing, startup/shutdown, monitor integration, and forwarding of flags from the parent to the child process. They do **not** measure actual model performance.

## 10. Known limitations

- Genuine full VLM execution still requires the team-supplied YOLO/CLIP/MLP/RAG adapters, trained alignment, images, and weights.
- The folder intentionally contains **no pretrained model weights** and cannot download them without network access.
- Nsight stage NVTX ranges are **CPU-side intervals**, not automatically pure GPU execution times; use CUDA GPU tracks for kernel durations.
- This package uses the `NsightBuffer` CPU/RAM sampler in all three separate experiment modes, so the latency category is still **instrumented** rather than a profiler-free baseline.
- `tegrastats` temperature and power measurements are system-level, sampled signals. No device validation has yet been performed for this packaged integration.
- GPU hardware counters and native CPU profiling depend on the board, Nsight release, and system permissions.

For exact code edits and the rationale, read **CHANGES.md**.
