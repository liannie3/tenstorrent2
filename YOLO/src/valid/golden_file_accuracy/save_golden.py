import sys, os
import argparse #cli arguments
from pathlib import Path
import torch

#PATH LOGIC
HARNESS_DIR = Path(__file__).parent.parent.parent / "substage_compiled" #NOTE: harness that produces the detections, must match the folder name
sys.path.insert(0, str(HARNESS_DIR)) #lets this file import main.py and config.py from the harness folder
os.chdir(HARNESS_DIR) #run from the harness folder so relative paths like the yolov8n.pt weights still work

import config
from main import main, load_image_path

#NOTE: runs the harness once and saves every image's detections, keyed by image file name, to be compared later
#      keyed by name so a --step subset can be checked against a full (step 1) golden, image by image
#USAGE: python save_golden.py --name jetson --precision fp32                 full set reference (step 1)
#       python save_golden.py --name jetson --precision bf16 --step 4        every 4th image
#       python save_golden.py --name jetson --precision fp32 --batch_size 8 --optimization eager
parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--name", default = "jetson", help = "platform tag for the golden file's name (defaulted to jetson)")
parser.add_argument("--precision", choices = ["fp32", "fp16", "bf16"], default = "fp32", help = "determines data type the harness runs in (defaulted to fp32, the reference)")
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1, the full set)")
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64], default = 1, help = "determines throughput of images sent through the model per batch (defaulted to 1)")
parser.add_argument("--optimization", choices = ["inductor", "aot_eager", "eager"], default = "inductor", help = "determines torch.compile backend (defaulted to inductor)")
args = parser.parse_args()
dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16} #will be run in either (fp32, fp16, bf16)

#temporarily override the config.py file's settings for this run only (same as the harness main.py does from its own args)
#NOTE: for the subset consistency check keep everything but --step the same on both files
config.DTYPE = dtypes[args.precision]
config.COMPILE_BACKEND = args.optimization
config.OPTIMIZATION = args.optimization
config.STEP = args.step

#stops GPU from quietly using a lower TF32 precision for small operations
if args.precision == "fp32" and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False #disallow TF32 matmuls
    torch.backends.cudnn.allow_tf32 = False       #disallow TF32 convolutions

results = main("profile", config.IMG_DIR, args.batch_size, args.step) #one list of per image detections per batch

#flatten batches into one detection tensor per image (rows: x1, y1, x2, y2, confidence, class)
dets = [d.float().cpu() for batch in results for d in batch]
#the harness runs img_paths[::step] in order and only drops the last partial batch, so detection i belongs to name i
names = [p.name for p in load_image_path(config.IMG_DIR)[::args.step]][:len(dets)]

#ensure directory for golden to be saved to exists
out_dir = config.BENCH_DIR / "golden_files"
out_dir.mkdir(parents = True, exist_ok = True)
#only non default options go in the name, e.g. jetson_bf16_step4, jetson_fp32_b8, jetson_fp32_eager
tag = (f"{args.name}_{args.precision}" + (f"_step{args.step}" if args.step != 1 else "") + (f"_b{args.batch_size}" if args.batch_size != 1 else "")
       + (f"_{args.optimization}" if args.optimization != "inductor" else ""))
out_path = out_dir / f"golden_{tag}.pt"
torch.save({"names": names, "dets": dets, "step": args.step, "batch_size": args.batch_size, "precision": args.precision,
            "optimization": args.optimization}, out_path)
print(f"saved golden file at {out_path} ({len(dets)} images)") #indicates where the golden file was saved, and what name it's saved under
print(f"compare with: python compare.py <reference tag> {tag}")
