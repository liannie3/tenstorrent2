import argparse
from pathlib import Path

import torch
import ttnn
from PIL import Image

from device import open_ttnn, close_ttnn, get_device
from tokenizer import download_model, download_tokenizer
from weights import convert_model_to_ttnn
from preprocess import preprocess_image
from embeddings import CLIP

# Run from anywhere:  python test.py
# Other images:       python test.py --images ~/images/a.jpg ~/images/b.jpg --texts "a photo of a car" "a photo of a tree"
MODEL = "openai/clip-vit-large-patch14-336"     # Updated model
HERE = Path(__file__).resolve().parent          # the tt_clip folder

parser = argparse.ArgumentParser()
parser.add_argument("--images", nargs="+", default=[str(HERE / "cat.jpg"), str(HERE / "dog.jpg")],
                    help="image files (default: cat.jpg and dog.jpg next to this script)")
parser.add_argument("--texts", nargs="+", default=["a photo of a cat", "a photo of a dog"])
args = parser.parse_args()

image_paths = [Path(p).expanduser() for p in args.images]
missing = [str(p) for p in image_paths if not p.is_file()]
if missing:
    raise SystemExit("Image(s) not found: " + ", ".join(missing))

# Tokenize once; padding to 77 matches CLIP's context length (pad token = end-of-text)
tokenizer = download_tokenizer(MODEL)
ids = tokenizer(args.texts, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt")["input_ids"]

hf_model = download_model(MODEL)                # CPU reference model, also the source of the weights

open_ttnn()                                     # must come before weight conversion
try:
    clip = CLIP(convert_model_to_ttnn(hf_model.state_dict()))
    tokens_tt = ttnn.from_torch(ids, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=get_device())

    for path in image_paths:
        pixels = preprocess_image(Image.open(path), 336).unsqueeze(0)    # [1, 3, 336, 336]
        image_tt = ttnn.from_torch(pixels, dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=get_device())

        logits_per_image, _ = clip.forward(image_tt, tokens_tt)
        tt_probs = ttnn.to_torch(ttnn.softmax(logits_per_image, dim=-1)).float()

        with torch.no_grad():
            ref_probs = hf_model(input_ids=ids, pixel_values=pixels).logits_per_image.softmax(-1)

        print(f"\n{path.name}")
        for i, text in enumerate(args.texts):
            print(f"  {text!r:40}  ttnn={tt_probs[0, i]:.3f}  hf={ref_probs[0, i]:.3f}")
finally:
    close_ttnn()