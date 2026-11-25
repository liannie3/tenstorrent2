"""Step 1: run every substage on the card and check it against the PyTorch CPU reference.

For one batch (or --verify-batches N) it compares, with Pearson correlation (PCC):
  pre_processing   the normalized input tensor the card built in D_normalize_layout vs ultralytics-style CPU preprocessing
  A_backbone       every layer 0-9 (C2f/SPPF outputs included)
  B_neck           every layer 10-21
  C_head           decoded boxes (xywh) and class probabilities, separately
  post_processing  final detections after NMS + rescale: how many CPU detections the card reproduces (same class, IoU >= 0.5)
"""

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

import config
import results
import timing
from stages import postprocess_stage, preprocess_stage
from tt_device import describe


def pcc(a, b):
    """Pearson correlation of two tensors (1.0 = identical up to scale/offset)."""
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    if a.numel() != b.numel():
        raise ValueError(f"size mismatch {a.numel()} vs {b.numel()}")
    if torch.equal(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float("nan")


def status_of(value):
    if value != value:  #nan
        return "FAIL"
    return "PASS" if value >= config.PCC_PASS else ("WARN" if value >= config.PCC_WARN else "FAIL")


def reference_input(imgs, input_hw):
    """Same letterbox as the TT path, then ultralytics' own CPU preprocess (BGR->RGB, HWC->CHW, /255) in fp32."""
    letterbox = LetterBox(new_shape=input_hw, auto=False, stride=32)
    x = np.stack([letterbox(image=im) for im in imgs])
    x = np.ascontiguousarray(x[..., ::-1].transpose(0, 3, 1, 2))
    return torch.from_numpy(x).float() / 255.0


@torch.inference_mode()
def reference_forward(torch_model, x):
    """The CUDA version's layer walk on the fp32 CPU model, keeping every layer's output."""
    capture, y = {}, []
    for m in torch_model.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        capture[m.i] = x
        y.append(x if m.i in torch_model.save else None)
    pred = x[0] if isinstance(x, (tuple, list)) else x  #Detect returns (predictions, raw outputs)
    return pred, capture


def tt_forward(tt_model, x):
    """Same walk on the card, one layer at a time, so a failing op is reported with its layer index and type."""
    capture, y = {}, []
    for m in tt_model.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        try:
            x = m(x)
        except Exception as e:
            raise RuntimeError(f"TT layer {m.i} ({m.type}) failed: {e}") from e
        capture[m.i] = x
        y.append(x if m.i in tt_model.save else None)
    return x, capture


def match_detections(tt, ref, iou_thr=0.5):
    """Greedy one-to-one matching of TT detections to CPU detections (same class, IoU >= iou_thr). Returns matched IoUs."""
    if len(tt) == 0 or len(ref) == 0:
        return []
    ious = torchvision.ops.box_iou(ref[:, :4], tt[:, :4]) * (ref[:, 5:6] == tt[:, 5].unsqueeze(0))
    used, matched = set(), []
    for r in torch.argsort(ref[:, 4], descending=True).tolist():
        cand = ious[r].clone()
        if used:
            cand[list(used)] = -1.0
        j = int(torch.argmax(cand))
        if cand[j] >= iou_thr:
            used.add(j)
            matched.append(float(cand[j]))
    return matched


def draw(img, dets, names, color=(0, 200, 255)):
    out = img.copy()
    for x1, y1, x2, y2, conf, cls in dets.tolist():
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(out, f"{names[int(cls)]} {conf:.2f}", (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    return out


@torch.inference_mode()
def verify(tt_model, torch_model, images, batch_size, ctx, input_hw=None, img_names=None, save_images=False,
           num_batches=1):
    input_hw = tuple(input_hw or config.INPUT_HW)
    hw_str = f"{input_hw[0]}x{input_hw[1]}"
    decode = "device" if tt_model.model[-1].decode_on_device else "host"
    dev = describe(ctx.device)
    stamp = datetime.now().isoformat(timespec="seconds")
    base = dict(timestamp=stamp, device=dev, batch_size=batch_size, input_hw=hw_str, decode=decode, **ctx.settings())
    rows, worst = [], "PASS"
    config.SUBSTEP_TIMING = False #no timing here, just correctness

    def record(check, value, st=None):
        nonlocal worst
        st = st or status_of(value)
        worst = {"PASS": worst, "WARN": "WARN" if worst == "PASS" else worst, "FAIL": "FAIL"}[st]
        rows.append(dict(base, check=check, pcc=f"{value:.5f}", status=st))
        print(f"  {check:<44} {value:9.5f}  {st}")

    n_batches = max(1, min(num_batches, len(images) // batch_size))
    print(f"VERIFY: TT ({dev}, act_mem={ctx.act_mem}, weights={ctx.weights_dtype_name}, fidelity={ctx.settings()['math_fidelity']}, "
          f"decode={decode}) vs PyTorch fp32 CPU, input {hw_str}, batch {batch_size}, {n_batches} batch(es)")
    det_tt_total = det_ref_total = det_matched = 0
    matched_ious = []

    for b in range(n_batches):
        imgs = images[b * batch_size : (b + 1) * batch_size]
        names = img_names[b * batch_size : (b + 1) * batch_size] if img_names else [f"img{b * batch_size + k}" for k in range(len(imgs))]

        # ---------------- Tenstorrent: every substage on the card
        t0 = time.perf_counter()
        try:
            im, shapes = preprocess_stage(imgs, ctx, input_hw)
        except Exception as e:
            raise RuntimeError(f"pre_processing failed in {timing.current['step']}: {e}") from e
        head, tt_layers = tt_forward(tt_model, im)
        try:
            tt_pred = head.assemble(head.to_host())
            tt_dets = postprocess_stage(head, shapes, ctx, input_hw, config.CONF_THRES, config.IOU_THRESH)
        except Exception as e:
            raise RuntimeError(f"post_processing failed in {timing.current['step']}: {e}") from e
        ctx.sync()
        elapsed = time.perf_counter() - t0
        if b == 0:
            print(f"  first TT pass took {elapsed:.1f} s (includes JIT kernel compiles and weight preparation)")

        # ---------------- PyTorch fp32 CPU reference
        x_ref = reference_input(imgs, input_hw)
        ref_pred, ref_layers = reference_forward(torch_model, x_ref)
        ref_dets = non_max_suppression(ref_pred, config.CONF_THRES, config.IOU_THRESH, max_det=300)
        for p, shape in zip(ref_dets, shapes):
            if len(p):
                p[:, :4] = scale_boxes(input_hw, p[:, :4], shape)

        # ---------------- compare
        print(f"  batch {b}: {', '.join(names)}")
        print(f"  {'check':<44} {'PCC':>9}  status")
        #the card's input is still in 0-255 units (the /255 lives in the stem conv); PCC can't see a scale error, so the
        #input must match exactly (0-255 integers are exact in bf16)
        tt_in = im.to_torch_nchw()[:, :3] / 255.0
        in_err = float((tt_in - x_ref).abs().max())
        in_status = "PASS" if in_err < 1e-6 else ("WARN" if in_err <= 1.5 / 255 else "FAIL")
        record("pre_processing: normalized input (exact)", pcc(tt_in, x_ref), in_status)
        if in_err >= 1e-6:
            print(f"    input differs from the CPU reference by up to {in_err * 255:.2f} pixel levels (expected 0)")
        for layer in tt_model.model:
            if layer.type == "Detect":
                continue
            stage = "A_backbone" if layer.i <= config.BACKBONE_BOUNDARY else "B_neck"
            record(f"{stage}: layer {layer.i:>2} {layer.type}", pcc(tt_layers[layer.i].to_torch_nchw(), ref_layers[layer.i]))
        record("C_head: boxes (xywh, pixels)", pcc(tt_pred[:, :4], ref_pred[:, :4]))
        record("C_head: class probabilities", pcc(tt_pred[:, 4:], ref_pred[:, 4:]))

        for k, (t, r) in enumerate(zip(tt_dets, ref_dets)):
            m = match_detections(t, r)
            det_tt_total, det_ref_total, det_matched = det_tt_total + len(t), det_ref_total + len(r), det_matched + len(m)
            matched_ious += m
            miou = f", mean IoU {np.mean(m):.3f}" if m else ""
            print(f"  post_processing: {names[k]}: TT {len(t)} dets, CPU {len(r)} dets, {len(m)} matched{miou}")
            if save_images:
                config.VERIFY_IMG_DIR.mkdir(parents=True, exist_ok=True)
                out = config.VERIFY_IMG_DIR / f"{Path(str(names[k])).stem}_tt.jpg"
                cv2.imwrite(str(out), draw(imgs[k], t, tt_model.names))

    recall = det_matched / det_ref_total if det_ref_total else 1.0
    precision = det_matched / det_tt_total if det_tt_total else 1.0
    rows.append(dict(base, check="post_processing: detections reproduced (matched / CPU)", pcc=f"{recall:.5f}",
                     status="PASS" if recall >= 0.9 else "WARN"))
    rows.append(dict(base, check="post_processing: TT detections confirmed (matched / TT)", pcc=f"{precision:.5f}",
                     status="PASS" if precision >= 0.9 else "WARN"))
    results.store_verify(rows)
    miou = f", mean IoU of matches {np.mean(matched_ious):.3f}" if matched_ious else ""
    print(f"  detections: CPU {det_ref_total}, TT {det_tt_total}, matched {det_matched} "
          f"({100 * recall:.1f}% of CPU, {100 * precision:.1f}% of TT){miou}")
    if save_images:
        print(f"  annotated TT detections written to {config.VERIFY_IMG_DIR}")
    print(f"VERIFY RESULT: {worst}  (PCC pass >= {config.PCC_PASS}, warn >= {config.PCC_WARN}); rows appended to {config.VERIFY_CSV_PATH}")
    return worst
