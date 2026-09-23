import torch
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor

name = "openai/clip-vit-large-patch14-336"
model = CLIPVisionModel.from_pretrained(name).eval()
proc = CLIPImageProcessor.from_pretrained(name)

img = Image.new("RGB", (1600, 900), "gray")   # swap in a real photo later
x = proc(images=img, return_tensors="pt")["pixel_values"]
print("input:", x.shape)                        # [1, 3, 336, 336]

with torch.no_grad():
    out = model(x, output_hidden_states=True)

print("num hidden states:", len(out.hidden_states))   # 25 = embeddings + 24 layers
print("layer 23:", out.hidden_states[-2].shape)       # [1, 577, 1024]
print("final:", out.last_hidden_state.shape)          # [1, 577, 1024]