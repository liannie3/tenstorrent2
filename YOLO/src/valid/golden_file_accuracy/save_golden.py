import sys, os
import argparse #cli arguments
from pathlib import Path
import torch
import torch_xla.runtime as xr #used to pick the bh p150a card as the xla (i.e. accelerated linear algebra) device
import torch_xla #used to set the compiler options (optimization level)

#PATH LOGIC
HARNESS_DIR = Path(__file__).parent.parent.parent / "xla_yolo" #NOTE: harness that produces the detections, must match the folder name
sys.path.insert(0, str(HARNESS_DIR)) #lets this file import main.py and config.py from the harness folder
os.chdir(HARNESS_DIR) #run from the harness folder so relative paths like the yolov8n.pt weights still work

import config
from main import main, load_image_path

#NOTE: runs the harness once and saves every image's detections, keyed by image file name, to be compared later
#      keyed by name so a --step subset can be checked against a full (step 1) golden, image by image
#      always bf16 (the card's native format), bf16 stays in the file name so tags line up with the CUDA golden files
#USAGE: python save_golden.py --name tt                                full set reference (step 1)
#       python save_golden.py --name tt --step 4                       every 4th image
#       python save_golden.py --name tt --batch_size 8 --optimization 1
parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--name", default = "tt", help = "platform tag for the golden file's name (defaulted to tt)")
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1, the full set)")
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64], default = 1, help = "determines throughput of images sent through the model per batch (defaulted to 1)")
parser.add_argument("--optimization", type = int, choices = [0, 1, 2], default = 0, help = "determines level of TT-XLA compiler optimization (defaulted to 0)")
args = parser.parse_args()

#temporarily override the config.py file's settings for this run only (same as the harness main.py does from its own args)
#NOTE: for the subset consistency check keep everything but --step the same on both files
config.DTYPE = torch.bfloat16
config.OPTIMIZATION = args.optimization
config.STEP = args.step

#NOTE: the harness main.py only picks the card and sets the optimization level under if __name__ == "__main__",
#      which doesn't run when main() is imported, so both are done here before main() loads the model
xr.set_device_type("TT") #pytorch xla defaults to run on host as device, so select TT card as device before model
torch_xla.set_custom_compile_options({"optimization_level": args.optimization})

results = main("profile", config.IMG_DIR, args.batch_size, args.step) #one list of per image detections per batch

#flatten batches into one detection tensor per image (rows: x1, y1, x2, y2, confidence, class)
dets = [d.float().cpu() for batch in results for d in batch]
#the harness runs img_paths[::step] in order and only drops the last partial batch, so detection i belongs to name i
names = [p.name for p in load_image_path(config.IMG_DIR)[::args.step]][:len(dets)]

#ensure directory for golden to be saved to exists
out_dir = config.BENCH_DIR / "golden_files"
out_dir.mkdir(parents = True, exist_ok = True)
#only non default options go in the name, e.g. tt_bf16_step4, tt_bf16_b8, tt_bf16_opt1
tag = (f"{args.name}_bf16" + (f"_step{args.step}" if args.step != 1 else "") + (f"_b{args.batch_size}" if args.batch_size != 1 else "")
       + (f"_opt{args.optimization}" if args.optimization != 0 else ""))
out_path = out_dir / f"golden_{tag}.pt"
torch.save({"names": names, "dets": dets, "step": args.step, "batch_size": args.batch_size, "precision": "bf16",
            "optimization": args.optimization}, out_path)
print(f"saved golden file at {out_path} ({len(dets)} images)") #indicates where the golden file was saved, and what name it's saved under
print(f"compare with: python compare.py <reference tag> {tag}")
