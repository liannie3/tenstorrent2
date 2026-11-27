import sys
import subprocess
from pathlib import Path
import argparse

from helper import save
#NOTE: GOAL is to show that running a subset of the image database (--step) doesn't change the predictions:
#      every image in a step 2 / 4 / 8 / 16 subset must get the same detections as the same image in the full set (step 1)
#      1. save_golden.py --step 1 once (the reference, all images)
#      2. save_golden.py --step N for every other step size
#      3. compare.py reference vs each subset, matched image by image by file name
#      everything but --step stays the same between the files, so the detections should be identical
#      (tiny differences are possible from GPU run to run nondeterminism, hence --accuracy / --max_conf_diff)
#USAGE: python main.py                                   all steps, fp32, batch 1, inductor
#       python main.py --precision bf16 --steps 4 16

parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--steps", type = int, nargs = "+", choices = [2,4,8,16], default = [2,4,8,16], help = "subset step sizes to check against the full set (defaulted to 2 4 8 16)")
parser.add_argument("--precision", choices = ["fp32", "fp16", "bf16"], default = "bf16", help = "data type of every run (defaulted to bf16)")
parser.add_argument("--optimization", choices = ["inductor", "aot_eager", "eager"], default = "inductor", help = "determines torch.compile backend (defaulted to inductor)")
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64], default = 1, help = "determines throughput of images sent through the model per batch (defaulted to 1)")
parser.add_argument("--accuracy", type = float, default = 99, help = "determines percentage of full set boxes that must be found on the same subset image (defaulted to 99%%)")
parser.add_argument("--max_conf_diff", type = float, default = 0.001, help = "determines max average confidence difference on matched boxes (defaulted to 0.001)")
args = parser.parse_args()

GOLDEN_SCRIPTS = Path(__file__).parent.parent / "golden_file_accuracy" #save_golden.py and compare.py, this file is in YOLO/src/valid/subset_consistency
NAME = "subset" #golden files are saved as golden_subset_<precision>..., separate from the platform goldens
common = ["--name", NAME, "--precision", args.precision, "--optimization", args.optimization, "--batch_size", str(args.batch_size)]

#CAPTURE RESULTS
print("step 1 (full set reference)")
reference = save(1)
rows, failed = [], [] #one row per step, steps that failed validation
for step in args.steps:
    print(f"step {step}")
    tag = save(step, GOLDEN_SCRIPTS)
    result = subprocess.run([sys.executable, "compare.py", reference, tag, "--accuracy", str(args.accuracy), "--max_conf_diff", str(args.max_conf_diff)],
                            cwd = GOLDEN_SCRIPTS, capture_output = True, text = True)
    #compare.py prints lines like:
    #  comparing 101 images (test is a subset of the 404 reference images)
    #  reference boxes also found in test: 812 / 812 = 100.0 %
    #  confidence difference on matched boxes: average 0.0   max 0.0
    #  detections per image: subset 8.04   full set 7.90
    row = {"step": step, "images": "-", "found %": "-", "avg conf diff": "-", "max conf diff": "-", "dets/img subset": "-", "dets/img full": "-"}
    for line in result.stdout.splitlines():
        words = line.split()
        if line.startswith("comparing"):
            row["images"] = words[1]
        elif line.startswith("reference boxes also found"):
            row["found %"] = words[-2]
        elif line.startswith("confidence difference"):
            row["avg conf diff"], row["max conf diff"] = words[6], words[-1]
        elif line.strip().startswith("detections per image"):
            row["dets/img subset"], row["dets/img full"] = words[4], words[7]
    if result.returncode != 0 or "FAILED" in result.stdout or row["found %"] == "-":
        failed.append(step)
        if result.returncode != 0 or row["found %"] == "-": #crashed rather than failed a check
            print(result.stdout[-3000:]); print(result.stderr[-3000:])
    rows.append(row)

#DISPLAY RESULTS
print(f"\n  {args.precision}, batch {args.batch_size}, {args.optimization}: every subset vs the full set (step 1), same images matched by file name")
keys = list(rows[0].keys())
print("  " + " ".join(f"{k:>15}" for k in keys))
for row in rows:
    print("  " + " ".join(f"{str(row[k]):>15}" for k in keys))

#VALIDATE RESULTS
if failed:
    print(f"\n    ===Subset consistency validation check FAILED!\n    ===Expected >= {args.accuracy}% of boxes found and average confidence difference <= {args.max_conf_diff}, failed steps: {', '.join(map(str, failed))}")
else:
    print(f"\n    ===Subset consistency validation check PASSED!\n    ===Every step {', '.join(map(str, args.steps))} subset image has the same detections as in the full set (>= {args.accuracy}% found, conf diff <= {args.max_conf_diff})")
