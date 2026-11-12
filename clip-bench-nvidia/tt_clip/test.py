import argparse
import time
from pathlib import Path

import torch
import ttnn
from PIL import Image

from device import open_ttnn, close_ttnn, get_device
from tokenizer import download_model            # returns CLIPVisionModel
from weights import convert_model_to_ttnn
from preprocess import preprocess_image
from embeddings import TtClipVisionModel

# Run from anywhere:  python test.py
# Other images:       python test.py --images ~/images/a.jpg ~/images/b.jpg ~/images/c.jpg
MODEL = "openai/clip-vit-large-patch14-336"
IMAGE_SIZE = 336
NUM_LAYERS = 23                                 # first 23 of 24 layers (LLaVA select_layer = -2)
HERE = Path(__file__).resolve().parent          # the tt_clip folder

parser = argparse.ArgumentParser()
parser.add_argument("--images", nargs="+", default=[str(HERE / "cat.jpg"), str(HERE / "dog.jpg")],
                    help="image files (default: cat.jpg and dog.jpg next to this script)")
parser.add_argument("--warmup", type=int, default=1, help="untimed runs on the first image")
parser.add_argument("--iters", type=int, default=5, help="timed runs on the first image")
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
    model = TtClipVisionModel(convert_model_to_ttnn(hf_model.state_dict()), num_layers=NUM_LAYERS)

    def to_tt(pixels):
        return ttnn.from_torch(pixels, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=get_device())

    # Correctness: one forward per image, compared against HF on CPU
    for path in image_paths:
        pixels = load_pixels(path)
        tt_out = ttnn.to_torch(model.forward(to_tt(pixels))).float()

        with torch.no_grad():
            # hidden_states[-2] = output after 23 layers; [:, 1:] drops CLS -> [1, 576, 1024]
            ref = hf_model(pixel_values=pixels, output_hidden_states=True).hidden_states[-2][:, 1:]

        tt_out = tt_out.reshape(ref.shape)
        print(f"{path.name}: PCC={pcc(tt_out, ref):.5f}  shape={tuple(tt_out.shape)}")

    # Timing: first image only, after warmup (warmup absorbs weight prep + kernel compile)
    image_tt = to_tt(load_pixels(image_paths[0]))

    for _ in range(args.warmup):
        ttnn.deallocate(model.forward(image_tt))
    ttnn.synchronize_device(get_device())

    times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        out = model.forward(image_tt)
        ttnn.synchronize_device(get_device())   # TT-NN ops are async
        times.append(time.perf_counter() - start)
        ttnn.deallocate(out)

    print(f"\nlatency over {args.iters} runs: mean={1000 * sum(times) / len(times):.1f} ms  "
          f"min={1000 * min(times):.1f} ms")
finally:
    close_ttnn()