# YOLOv8n stage-by-stage benchmark on Tenstorrent Blackhole (p150a)

This is a TT-NN port of the CUDA/Jetson `yoloDIY_test.py` benchmark, laid out like its modular version (`config.py`, `timing.py`,
`stages.py`, `run.py`, `results.py`, `main.py`). The substages stay the same: **preprocess** (A–D) → **inference**
(A_backbone / B_neck / C_head) → **postprocess**. The model itself runs on the card through TT-NN, the Python op library in
tt-metal.

| step | what | status |
|---|---|---|
| 1 | every substage runs on the card; `verify` checks each one against PyTorch on the CPU | **ready** |
| 2 | latency: wall pass, substep pass, end-to-end pass → `benchmarking/stage_timings_tt.csv` | **ready** |
| 3 | hardware metrics: TT telemetry (power, temperature, clocks) replacing tegrastats / nvidia-smi | next |
| 4 | profiling: Tracy + the device profiler (on-card per-op times, the CUDA-event equivalent) | later |
| 5 | optimisation: L1 sharding, trace capture + 2 command queues | later |

## Setup (on the machine with the p150a)

1. **tt-metal / TT-NN.** Use either a source build (activate `python_env`) or the `tt-metalium` container. Check it works:
   `python -c "import ttnn; d = ttnn.open_device(device_id=0); print(d.arch()); ttnn.close_device(d)"`.
   The code is written against the tt-metal API as of v0.81–v0.83 (Nov 2026). On much older releases, some conv2d / pool
   keyword arguments differ.
2. **ultralytics, without letting pip touch tt-metal's torch / numpy<2 pins:**
   ```bash
   pip install --no-deps ultralytics
   ```
   That's enough. With ultralytics 8.4, everything this code uses (loading/fusing `yolov8n.pt`, `LetterBox`, NMS, `scale_boxes`)
   imports with only numpy, torch, torchvision, opencv and pillow, all of which tt-metal's environment already has. (A plain
   `pip install ultralytics` also pulls `opencv-python`, which clashes with tt-metal's `opencv-python-headless`.) If an import
   error ever names a missing module, `pip install` just that module.
3. Put this folder next to your CUDA version, at the same depth, so `benchmarking/` and `input_images/CAM_FRONT/` resolve the same
   way. Otherwise pass `--img-dir`. `yolov8n.pt` downloads automatically, or you can copy it into the working directory.

## Step 1 — run the substages on the card and verify them

```bash
python main.py verify 1 --num-images 4 --save-images
```

* The first run JIT-compiles every kernel. That can take a few minutes; later runs read from the on-disk cache.
* For each batch it prints a PCC (Pearson correlation vs the fp32 CPU model) for the normalized input, **every layer 0–21**
  (tagged A_backbone / B_neck), the decoded boxes and class probabilities (C_head), and then how many of the CPU's detections
  the card reproduces after NMS. Rows also go to `benchmarking/verify_tt.csv`. `--save-images` draws the card's detections
  into `benchmarking/tt_verify/`.
* Expected results, from running the same code with bf16 rounding emulated on the CPU (`tests/run_cpu_check.py`): every layer
  PCC ≥ 0.999, boxes ≈ 0.99999, class probabilities ≥ 0.999, 100% of detections reproduced, mean box IoU ≈ 0.986.
* If something fails, the message names the layer (`TT layer 12 (C2f) failed: ...`) or the substep. Send me that line and
  the tt-metal version, and we fix that op before moving on.

## Step 2 — latency

```bash
python main.py latency 1          # batch sizes 1 2 4 8 16 as before
```

The command runs three passes. Every pass starts with 2 warmup iterations, and timings are per image (median per batch ÷ batch):

| pass | sync | what it measures |
|---|---|---|
| `wall_timing` | between stages | ultralytics-style stage wall times (same as the CUDA pass 1) |
| `substep_timing` | around every device substep | where time goes inside a stage |
| `e2e` | none (only the blocking device→host read) | real loop latency, where host dispatch overlaps the card; also prints images/s |

Settings worth sweeping once the default verifies (each one is recorded in the CSV):

* `--act-mem l1`: activations in L1 instead of DRAM. This should be faster; it fits batch 1–2 at 640×640.
* `--math-fidelity lofi`, `--no-fp32-acc`, `--weights-dtype bfp8`: faster math at some accuracy cost. bfp8 noticeably lowers
  yolov8n's class-score PCC, so re-run `verify` with it.
* `--input-hw 384 640`: the shape your CUDA run actually used (`auto=True` letterbox of a 1600×900 frame). Use it for an
  apples-to-apples comparison with the GPU numbers.
* `--decode host`: box decode on the CPU in fp32 (adds a `A2_decode_host` postprocess substep).

## How the CUDA version maps to TT

| CUDA version | TT version |
|---|---|
| `model.to("cuda")`, NCHW tensors | `TtYoloV8`: each ultralytics block gets a TT counterpart. Activations are channels-last `[1, 1, N·H·W, C]`, the form `ttnn.conv2d` uses |
| `torch.cuda.synchronize()` | `ttnn.synchronize_device(device)` |
| CUDA events for async substep times | no timestamped events in TT-NN; substeps use sync + `perf_counter`. On-card op times come from the profiler (step 4) |
| NVTX ranges | Tracy zones / signposts (`--tracy`, run under `python -m tracy`) |
| `C_host_to_gpu` | `C_host_to_device` (`ttnn.from_torch(..., device=...)`) |
| `D_normalize_layout`: permute, flip, float, /255 on the GPU | flip, transpose and uint8→bf16 on the host (where ultralytics itself does them). On the card: pad 3→16 channels, NCHW→NHWC, tilize. The /255 is folded into the stem conv's weights |
| NMS on the GPU | `A_device_to_host`, then NMS and rescale on the CPU |
| fp16 / fp32 | bf16 activations; bf16 or bfp8 weights; math fidelity |
| `LetterBox(auto=True)` (dynamic shape) | `auto=False` to one static `--input-hw`: TT kernels are compiled per shape |

Other notes:
* `run_layers` is unchanged. The TT layers carry the same `i` / `f` attributes, and the model carries the same `save` list,
  so the 0–9 / 10–21 / 22 split works as before.
* `C2f.cv1(x).chunk(2)` runs as two half-width convs (identical math). This avoids slicing a tiled tensor at 16 channels,
  and it's what tt-metal's own YOLOv8 port does.
* The Detect decode runs on the card without a 16-wide softmax. It uses exp → two small matmuls (Σe and Σk·e) → divide →
  one matmul for dist2bbox → + anchors, plus sigmoid on the classes. This is mathematically identical to ultralytics; box
  coordinates come back in bf16 (±1–2 px at 640).
* A trailing partial batch is dropped, because a different batch size would recompile mid-run.
* Eager TT-NN is host-dispatch bound: about 190 ops per image, each launched from Python. Trace capture (step 5) removes
  that, and it's how tt-metal's Blackhole YOLOv8s reaches 372 FPS.
* The CUDA `results.py` printed `TOTAL_WALL_TIME` per batch under a per-image header. Here it is divided by batch size like
  every other row.

## Checking the code without a card

```bash
python tests/run_cpu_check.py            # bf16-emulated; uses ultralytics' sample images or --img-dir
python tests/run_cpu_check.py --exact    # fp32: must match ultralytics (PCC 1.00000 everywhere)
```

`tests/fake_ttnn.py` stands in for `ttnn`, using torch on the CPU. It has the same function names and arguments, and it
enforces the layout rules the port relies on. It proves the wiring, shapes, decode math and harness are right. It can't prove
that the card accepts every op as called; `main.py verify` on the p150a does that.

## Files

| file | role |
|---|---|
| `config.py` | paths, model / TT device settings, run flags |
| `tt_device.py` | open/close the card, `TtContext` (memory placement, dtypes, compute configs, layout helpers) |
| `tt_yolo.py` | TT-NN YOLOv8: `TtConv`, `TtC2f`, `TtSPPF`, `TtUpsample`, `TtConcat`, `TtDetect`, `TtYoloV8` |
| `stages.py` | `preprocess_stage` / `inference_stage` / `postprocess_stage` and `run_layers` |
| `timing.py` | `timed()` (sync + perf_counter), Tracy markers |
| `run.py` | image loading, batching, warmup, `run_pipeline` |
| `results.py` | printing and CSV output |
| `verify.py` | step 1 checks |
| `main.py` | CLI |
| `tests/` | `fake_ttnn.py` + `run_cpu_check.py` |

Ultralytics YOLOv8 is AGPL-3.0, which is why tt-metal keeps its own YOLO ports off its main branch. This port uses
ultralytics the same way your CUDA version does.
