import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clipbench import config as cfg          # noqa: E402
from clipbench import data                   # noqa: E402
from clipbench import model as model_lib     # noqa: E402

# Run from anywhere:  python test_scripts/nvidia_parity.py
# Other images:       python test_scripts/nvidia_parity.py --images a.jpg b.jpg
# Same accuracy check as tt_clip/test.py: PCC of the NVIDIA encoder output vs fp32 CPU HF, per image.
ROOT = Path(__file__).resolve().parents[1]
TT_CLIP = ROOT / "tt_clip"

parser = argparse.ArgumentParser()
parser.add_argument("--profile", default="jetson", choices=cfg.available_profiles())
parser.add_argument("--images", nargs="+", default=[str(TT_CLIP / "cat.jpg"), str(TT_CLIP / "dog.jpg")],
                    help="image files (default: cat.jpg and dog.jpg from tt_clip)")
args = parser.parse_args()

image_paths = [Path(p).expanduser() for p in args.images]
missing = [str(p) for p in image_paths if not p.is_file()]
if missing:
    raise SystemExit("Image(s) not found: " + ", ".join(missing))


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


test_cfg = cfg.load_config(args.profile)
ref_cfg = copy.deepcopy(test_cfg)
ref_cfg["run"].update(device="cpu", precision="fp32")      # golden: fp32 on CPU

test_dev, test_dtype = test_cfg["run"]["device"], model_lib.resolve_dtype(test_cfg["run"]["precision"])
print(f"layer_index = {test_cfg['model']['layer_index']}  keep_cls = {test_cfg['model']['keep_cls']}")

processor = model_lib.load_processor(test_cfg)             # same image processor as the benchmark
test_model = model_lib.load_encoder(test_cfg)              # encoder under test (CUDA, config precision)
ref_model = model_lib.load_encoder(ref_cfg)                # CPU fp32 reference

print(f"hidden_act = {ref_model.config.hidden_act}")        # should be quick_gelu
print(f"comparing {test_dev} {test_cfg['run']['precision']} vs cpu fp32\n")

results = []
for path in image_paths:
    images = data.load_images([path])
    test_pixels = data.preprocess(processor, images, test_dev, test_dtype)
    ref_pixels = data.preprocess(processor, images, "cpu", torch.float32)

    with torch.inference_mode():
        test_out = model_lib.select_image_features(test_model(test_pixels, output_hidden_states=True), test_cfg).cpu()
        ref_out = model_lib.select_image_features(ref_model(ref_pixels, output_hidden_states=True), ref_cfg)

    results.append((pcc(test_out, ref_out), cosine(test_out, ref_out)))
    print(f"{path.name}: PCC={results[-1][0]:.5f}  cosine={results[-1][1]:.5f}  shape={tuple(test_out.shape)}")

print(f"\nmean over {len(results)} images: PCC={sum(r[0] for r in results) / len(results):.5f}  "
      f"cosine={sum(r[1] for r in results) / len(results):.5f}")
