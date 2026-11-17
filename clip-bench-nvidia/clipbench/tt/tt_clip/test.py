import argparse
import time
from pathlib import Path

import torch
import ttnn
from PIL import Image

from .device import open_ttnn, close_ttnn, get_device
from .loaders import download_model            # returns CLIPVisionModel
from .weights import convert_model_to_ttnn
from .preprocess import preprocess_image
from .embeddings import TTClipVisionModel

# Run from the project root:  python -m clipbench.tt.tt_clip.test
# Other images:               python -m clipbench.tt.tt_clip.test --images ~/images/a.jpg ~/images/b.jpg
MODEL = "openai/clip-vit-large-patch14-336"
IMAGE_SIZE = 336
NUM_LAYERS = 23                                 # first 23 of 24 layers (LLaVA select_layer = -2)
SAMPLES = Path(__file__).resolve().parents[2] / "common" / "sample_images"

parser = argparse.ArgumentParser()
parser.add_argument("--images", nargs="+", default=[str(SAMPLES / "cat.jpg"), str(SAMPLES / "dog.jpg")],
                    help="image files (default: cat.jpg and dog.jpg in clipbench/common/sample_images)")
parser.add_argument("--warmup", type=int, default=2, help="untimed warmup passes over all images")
parser.add_argument("--iters", type=int, default=20, help="timed frames (each frame runs every image once)")
parser.add_argument("--sweep", action="store_true", help="per-depth PCC sweep only, no timing")
args = parser.parse_args()

image_paths = [Path(p).expanduser() for p in args.images]
missing = [str(p) for p in image_paths if not p.is_file()]
if missing:
    raise SystemExit("Image(s) not found: " + ", ".join(missing))


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def load_pixels(path):
    return preprocess_image(Image.open(path), IMAGE_SIZE).unsqueeze(0)   # [1, 3, 336, 336]


hf_model = download_model(MODEL)                # CPU reference model, also the source of the weights
print(f"hidden_act = {hf_model.config.hidden_act}")   # should be quick_gelu

open_ttnn()                                     # must come before weight conversion
try:
    model = TTClipVisionModel(convert_model_to_ttnn(hf_model.state_dict()), num_layers=NUM_LAYERS)

    def to_tt(pixels):
        return ttnn.from_torch(pixels, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=get_device())

    if args.sweep:
        # Per-depth PCC against HF hidden_states, to localize accuracy loss. Skips timing.
        sd_tt = convert_model_to_ttnn(hf_model.state_dict())
        pixels = load_pixels(image_paths[0])
        with torch.no_grad():
            hs = hf_model(pixel_values=pixels, output_hidden_states=True).hidden_states
        for k in [0, 1, 2, 6, 12, 18, 23]:
            m = TTClipVisionModel(sd_tt, num_layers=k)
            out = ttnn.to_torch(m.forward(to_tt(pixels))).float().reshape(1, 576, 1024)
            print(f"after {k:2d} layers: PCC={pcc(out, hs[k][:, 1:]):.5f}")
        raise SystemExit

    # Correctness: one forward per image, compared against HF on CPU
    for path in image_paths:
        pixels = load_pixels(path)
        tt_out = ttnn.to_torch(model.forward(to_tt(pixels))).float()

        with torch.no_grad():
            # hidden_states[-2] = output after 23 layers; [:, 1:] drops CLS -> [1, 576, 1024]
            ref = hf_model(pixel_values=pixels, output_hidden_states=True).hidden_states[-2][:, 1:]

        tt_out = tt_out.reshape(ref.shape)
        print(f"{path.name}: PCC={pcc(tt_out, ref):.5f}  shape={tuple(tt_out.shape)}")

    # Timing: same loop structure as clipbench/tt/benchmark_tests/gpu_pure_latency.py.
    #   frame i -> for each view (image) -> [preprocess + upload, untimed]
    #                                       -> sync (empty queue) -> t0 -> encode -> sync -> t1
    # TT-NN has no device events, so every span is a perf_counter bracketed by synchronizes.
    views = len(image_paths)

    def sync():
        ttnn.synchronize_device(get_device())

    # Warmup: first forward pays weight prep + kernel compile. Timed separately, then discarded.
    warm_start = time.perf_counter()
    for _ in range(args.warmup):
        for path in image_paths:
            ttnn.deallocate(model.forward(to_tt(load_pixels(path))))
    sync()
    warm_ms = 1000 * (time.perf_counter() - warm_start)

    timings = []                                  # (stage, frame, view, ms)
    subtest_start = time.perf_counter_ns()

    for i in range(args.iters):
        iter_start = time.perf_counter_ns()       # Iteration: start

        for view, path in enumerate(image_paths):

            # Decode + preprocess + upload [CPU, untimed]
            pixel_values = to_tt(load_pixels(path))

            # Empty queue, so the clock starts when the card does
            sync()

            # Encode [device]
            t0 = time.perf_counter_ns()
            out = model.forward(pixel_values)
            sync()
            t1 = time.perf_counter_ns()

            timings.append(('encode', i, view, (t1 - t0) / 1e6))
            ttnn.deallocate(out)
            ttnn.deallocate(pixel_values)

        timings.append(('iter', i, -1, (time.perf_counter_ns() - iter_start) / 1e6))

    subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6

    def stats(vals):
        v = sorted(vals)
        return (f"mean={sum(v) / len(v):.1f} ms  median={v[len(v) // 2]:.1f} ms  "
                f"min={v[0]:.1f} ms  max={v[-1]:.1f} ms")

    print(f"\nwarmup ({args.warmup} x {views} images, includes compile): {warm_ms:.1f} ms")
    for view, path in enumerate(image_paths):
        print(f"encode [{path.name}] over {args.iters} frames: "
              f"{stats([ms for st, _, v, ms in timings if st == 'encode' and v == view])}")
    print(f"encode [all]  : {stats([ms for st, _, _, ms in timings if st == 'encode'])}")
    print(f"iter ({views} views): {stats([ms for st, _, _, ms in timings if st == 'iter'])}")
    n_images = args.iters * views
    print(f"subtest        : {subtest_ms:.1f} ms for {n_images} images "
          f"({1000 * n_images / subtest_ms:.2f} images/s, includes untimed preprocessing)")
finally:
    close_ttnn()
