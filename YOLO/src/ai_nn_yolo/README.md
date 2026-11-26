# YOLOv8n stage-by-stage benchmark on Tenstorrent Blackhole (p150a), TT-NN

The TT-NN model is **tt-metal's YOLOv8x demo ported to YOLOv8n**. The rest of the folder is the same harness layout as the
CUDA/Jetson and TT-XLA versions (`config.py`, `timing.py`, `stages.py`, `run.py`, `results.py`, `main.py`), with the same
substages: **preprocess** (A–D) → **inference** (A_backbone / B_neck / C_head) → **postprocess**.

## Where the model comes from

| file | origin |
|---|---|
| `ttnn_yolov8n.py` | `models/demos/yolov8x/tt/ttnn_yolov8x.py`, tenstorrent/tt-metal **v0.64.0** (Apache-2.0, © 2025 Tenstorrent AI ULC) |
| `ttnn_yolov8n_utils.py` | `models/demos/yolov8x/tt/ttnn_yolov8x_utils.py` (same release), plus two helpers from `models/experimental/yolo_common/yolo_utils.py` |

The yolov8x demo was removed from tt-metal's main branch after v0.64.0. Every change from the original is marked in the
code with `#NOTE: yolov8n port:`:

* **Layer configs.** Each `(kernel, stride, pad, out_ch, in_ch)` tuple uses yolov8n's widths (width multiple 0.25 vs
  1.25), and the bottleneck counts per C2f are yolov8n's (depth 0.33: 1, 2, 2, 1, then 1 in the neck; yolov8x has 3, 6, 6, 3
  and 3). The Detect head widths are 64/80 instead of 80/320.
* **Non-square input.** The original recovered height and width as `sqrt(N*H*W)`, which only works for 640×640 at batch 1.
  Every module now takes `(h, w)` explicitly, so 384×640 works.
* **Batch > 1.** `batch_size` is passed to every conv, and the Detect head's reshape/slice use it (they were hard-coded to 1).
* **Narrow layers.** Block sharding falls back to conv2d's own choice for layers narrower than 64 channels. yolov8n goes
  down to 16 channels, less than one tile per core column.
* **Numerics from config.** `WEIGHTS_DTYPE` (bfp8 / bf16) and `MATH_FIDELITY` (lofi / hifi2 / hifi4) come from
  `config.py`; the defaults are the demo's bfloat8_b + LoFi.
* **Normalization folded into the stem.** `/255` is folded into the stem conv's weights, so the card gets 0–255 pixels
  with no on-card multiply.
* **Stage split.** The forward is split into `prepare_input` / `backbone` / `neck` / `head`, so the harness can time each
  part the way the CUDA version does.
* **Built from the state dict.** The conv list in the preprocessor is built from the state dict instead of yolov8x's
  hand-written list.

## Run (on the machine with the p150a)

```bash
python main.py --setting verify                       # step 1: each stage on the card vs PyTorch on the CPU
python main.py --setting latency --batch_size 1       # step 2: wall / substep / end-to-end passes -> benchmarking/stage_timings_tt.csv
python main.py --setting holistic --batch_size 4      # both
```

* The first run JIT-compiles every kernel. That takes minutes on a cold cache; after that, kernels come from the disk cache.
* `verify` checks:
  * the input exactly;
  * the three backbone outputs (layers 4, 6, 9) and the three neck outputs (15, 18, 21) by PCC;
  * the head's boxes and class probabilities;
  * how many of the CPU's detections the card reproduces after NMS.

  Rows go to `benchmarking/verify_tt.csv`. The demo itself validated at PCC 0.978 with bfp8 + LoFi, so PCC ≥ 0.97 is a
  WARN, not a FAIL.
* **Batch size is capped by L1.** Like the demo, every conv uses `Conv2dL1FullSliceConfig`, so the whole feature map has
  to fit in on-chip SRAM. Start at batch 1 and go up (2, 4, 8, …) until it reports an L1 allocation error. That error is
  the limit, the same way TT-XLA optimization level 2 stopped at batch 16.

## Checking the code without a card

```bash
python tests/run_cpu_check.py                          # exact fp32: must match ultralytics (PCC 1.00000 everywhere)
FAKE_TTNN_DTYPE=bfp8 python tests/run_cpu_check.py     # bfloat8_b rounding, to estimate the accuracy cost
```

`tests/fake_ttnn.py` stands in for `ttnn`, using torch on the CPU. It checks shapes, the height/width/batch handed to every
conv, channel splits/concats, the box decode, and use-after-deallocate. It can't tell whether the card accepts a sharding
choice or fits a feature map in L1; `main.py --setting verify` on the p150a does that.

Last CPU check (yolov8n.pt, 384×640):
* **fp32, batch 1 and 2:** PCC 1.00000 at every check, 100% of detections reproduced.
* **bfp8 emulated, batch 2:** layers ≥ 0.997, boxes 0.99986, class probabilities 0.99384, 9/9 detections matched, mean
  IoU 0.97. LoFi isn't emulated, so expect the card to come in a little lower.

## Files

| file | role |
|---|---|
| `config.py` | paths, model / TT device settings, run flags |
| `tt_device.py` | open/close the card, `TtContext` (device + settings recorded in the csv) |
| `ttnn_yolov8n.py` | the ported TT-NN model: `TtConv`, `TtBottleneck`, `TtC2f`, `TtSppf`, `TtDetectCv2`, `TtDFL`, `TtDetect`, `TtDetectionModel` |
| `ttnn_yolov8n_utils.py` | BatchNorm folding, weight preprocessing, anchors, box decode, core-grid helpers |
| `stages.py` | `preprocess_stage` / `inference_stage` / `postprocess_stage` |
| `timing.py` | `timed()` (sync + perf_counter), Tracy markers |
| `run.py` | image loading, batching, warmup, `run_pipeline` |
| `results.py` | printing and CSV output |
| `verify.py` | step 1 checks |
| `main.py` | CLI |
| `tests/` | `fake_ttnn.py` + `run_cpu_check.py` |

## Licenses

* tt-metal (the yolov8x demo this model is ported from) is Apache-2.0. Keep the SPDX headers and the `#NOTE: yolov8n port:`
  markers, which list the changes as section 4(b) of the license requires.
* Ultralytics YOLOv8 (the model, the weights, LetterBox, NMS) is AGPL-3.0.
