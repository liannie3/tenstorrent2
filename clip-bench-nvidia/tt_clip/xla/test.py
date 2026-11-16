import argparse
import os
import sys
import time
from pathlib import Path

# Run from anywhere:  python xla/test.py            (or `python test.py` from inside xla/)
# Other images:       python test.py --images ~/images/a.jpg ~/images/b.jpg
# Max optimization:   python test.py --opt-level 2 --trace
#
# Same structure as tt_clip/test.py so the two are directly comparable:
# PCC vs HF fp32 on CPU per image, then a frames x views timing loop.
MODEL = "openai/clip-vit-large-patch14-336"
IMAGE_SIZE = 336
NUM_LAYERS = 23                                 # first 23 of 24 layers (LLaVA select_layer = -2)
HERE = Path(__file__).resolve().parent          # tt_clip/xla
TT_CLIP = HERE.parent                           # tt_clip, for the shared loaders and images

parser = argparse.ArgumentParser()
parser.add_argument("--images", nargs="+", default=[str(TT_CLIP / "cat.jpg"), str(TT_CLIP / "dog.jpg")],
                    help="image files (default: cat.jpg and dog.jpg in tt_clip/)")
parser.add_argument("--warmup", type=int, default=3,
                    help="untimed warmup passes over all images (TT-XLA docs: at least 3; "
                         "pass 1 compiles, pass 2 captures the trace)")
parser.add_argument("--iters", type=int, default=20, help="timed frames (each frame runs every image once)")
parser.add_argument("--opt-level", type=int, default=1, choices=[0, 1, 2],
                    help="TT-XLA optimization_level: 0 none, 1 recommended, 2 max SRAM")
parser.add_argument("--trace", action="store_true", help="enable TT-XLA runtime trace")
args = parser.parse_args()

# Must be set before torch_xla is imported (TT-XLA performance docs)
if args.trace:
    os.environ.setdefault("TT_RUNTIME_TRACE_REGION_SIZE", "10000000")   # ~10 MB

import torch                                    # noqa: E402
from PIL import Image                           # noqa: E402

sys.path.insert(0, str(TT_CLIP))                # tt_clip's loaders/preprocess (neither imports ttnn)
from loaders import download_model              # noqa: E402  returns CLIPVisionModel
from preprocess import preprocess_image         # noqa: E402
import clip_xla                                 # noqa: E402

image_paths = [Path(p).expanduser() for p in args.images]
missing = [str(p) for p in image_paths if not p.is_file()]
if missing:
    raise SystemExit("Image(s) not found: " + ", ".join(missing))


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def load_pixels(path):
    return preprocess_image(Image.open(path), IMAGE_SIZE).unsqueeze(0)   # [1, 3, 336, 336] fp32


hf_model = download_model(MODEL)                # CPU fp32 reference
print(f"hidden_act = {hf_model.config.hidden_act}")   # should be quick_gelu
print(f"optimization_level = {args.opt_level}  trace = {args.trace}")

clip_xla.open_tt()
# The XLA model gets its own copy: build() truncates layers and casts to bf16 in place
model, device = clip_xla.build(download_model(MODEL), NUM_LAYERS,
                               optimization_level=args.opt_level, enable_trace=args.trace)


def to_tt(pixels):
    return pixels.to(dtype=torch.bfloat16).to(device)


with torch.no_grad():
    # Correctness: one forward per image, compared against HF on CPU.
    # The first call also compiles the graph, so it is slow.
    for path in image_paths:
        pixels = load_pixels(path)
        tt_out = model(to_tt(pixels)).cpu().float()

        # hidden_states[-2] = output after 23 layers; [:, 1:] drops CLS -> [1, 576, 1024]
        ref = hf_model(pixel_values=pixels, output_hidden_states=True).hidden_states[-2][:, 1:]

        tt_out = tt_out.reshape(ref.shape)
        print(f"{path.name}: PCC={pcc(tt_out, ref):.5f}  shape={tuple(tt_out.shape)}")

    # Timing: same loop structure as tt_clip/test.py.
    #   frame i -> for each view (image) -> [preprocess + upload, untimed]
    #                                       -> wait (empty queue) -> t0 -> encode -> wait -> t1
    #                                       -> readback to CPU -> t2
    # 'encode' relies on xm.wait_device_ops() blocking on the TT device. 'encode+readback'
    # ends on a .cpu() copy, which cannot finish before the device does, so it is the
    # upper bound. If encode is near zero and encode+readback is not, the wait is not
    # blocking on this plugin and encode+readback is the number to trust.
    views = len(image_paths)

    warm_start = time.perf_counter()
    for _ in range(args.warmup):
        for path in image_paths:
            model(to_tt(load_pixels(path))).cpu()
    clip_xla.wait()
    warm_ms = 1000 * (time.perf_counter() - warm_start)

    timings = []                                # (stage, frame, view, ms)
    subtest_start = time.perf_counter_ns()

    for i in range(args.iters):
        iter_start = time.perf_counter_ns()     # Iteration: start

        for view, path in enumerate(image_paths):

            # Decode + preprocess + upload [CPU, untimed]
            pixel_values = to_tt(load_pixels(path))

            # Empty queue, so the clock starts when the card does
            clip_xla.wait()

            # Encode [device]
            t0 = time.perf_counter_ns()
            out = model(pixel_values)
            clip_xla.wait()
            t1 = time.perf_counter_ns()

            # Readback [device -> host]
            out.cpu()
            t2 = time.perf_counter_ns()

            timings.append(("encode", i, view, (t1 - t0) / 1e6))
            timings.append(("encode+readback", i, view, (t2 - t0) / 1e6))

        timings.append(("iter", i, -1, (time.perf_counter_ns() - iter_start) / 1e6))

    subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6


def stats(vals):
    v = sorted(vals)
    return (f"mean={sum(v) / len(v):.1f} ms  median={v[len(v) // 2]:.1f} ms  "
            f"min={v[0]:.1f} ms  max={v[-1]:.1f} ms")


def select(stage, view=None):
    return [ms for st, _, v, ms in timings if st == stage and (view is None or v == view)]


print(f"\nwarmup ({args.warmup} x {views} images, includes compile): {warm_ms:.1f} ms")
for view, path in enumerate(image_paths):
    print(f"encode [{path.name}] over {args.iters} frames: {stats(select('encode', view))}")
print(f"encode [all]  : {stats(select('encode'))}")
print(f"encode+readback [all]: {stats(select('encode+readback'))}")
print(f"iter ({views} views): {stats(select('iter'))}")
n_images = args.iters * views
print(f"subtest        : {subtest_ms:.1f} ms for {n_images} images "
      f"({1000 * n_images / subtest_ms:.2f} images/s, includes untimed preprocessing)")
