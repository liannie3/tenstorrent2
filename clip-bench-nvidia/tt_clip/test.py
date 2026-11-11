import argparse
import torch
import ttnn
from PIL import Image

from device import open_ttnn, close_ttnn, get_device
from tokenizer import download_model, download_tokenizer
from weights import convert_model_to_ttnn
from preprocess import preprocess_image
from embeddings import CLIP

# To run, python run_clip.py --images ~/images/cat.jpg ~/images/dog.jpg --texts "a photo of a cat" "a photo of a dog"
MODEL = "openai/clip-vit-base-patch32"

parser = argparse.ArgumentParser()
parser.add_argument("--images", nargs="+", required=True, help="paths to local image files")
parser.add_argument("--texts", nargs="+", default=["a photo of a cat", "a photo of a dog"])
args = parser.parse_args()

# Tokenize once; padding to 77 matches CLIP's context length (pad token = end-of-text)
tokenizer = download_tokenizer(MODEL)
ids = tokenizer(args.texts, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt")["input_ids"]

hf_model = download_model(MODEL)       # CPU reference model, also the source of the weights

open_ttnn()                            # must come before weight conversion
try:
    clip = CLIP(convert_model_to_ttnn(hf_model.state_dict()))
    tokens_tt = ttnn.from_torch(ids, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=get_device())

    for path in args.images:
        pixels = preprocess_image(Image.open(path), 224).unsqueeze(0)    # [1, 3, 224, 224]
        image_tt = ttnn.from_torch(pixels, dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=get_device())

        logits_per_image, _ = clip.forward(image_tt, tokens_tt)
        tt_probs = ttnn.to_torch(ttnn.softmax(logits_per_image, dim=-1)).float()

        with torch.no_grad():
            ref_probs = hf_model(input_ids=ids, pixel_values=pixels).logits_per_image.softmax(-1)

        print(f"\n{path}")
        for i, text in enumerate(args.texts):
            print(f"  {text!r:40}  ttnn={tt_probs[0, i]:.3f}  hf={ref_probs[0, i]:.3f}")
finally:
    close_ttnn()