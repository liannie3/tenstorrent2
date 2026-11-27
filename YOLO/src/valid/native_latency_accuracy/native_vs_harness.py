import sys
import subprocess
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--iterations", type = int, choices = [1,2,3,4,5,6,7,8,9,10],  default = 5, help = "determines how many iterations are run (defaulted to 5)") #used to allow user tuning of noise reducton
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1)") #used to create accurate subset of db
parser.add_argument("--accuracy", type = int, choices = [1,2,3,4,5,6,7,8,9,10],  default = 5, help = "determines the percent accuracy to validate against (defaulted to 5%%)") #used to allow user tuning accuracy threshold
args = parser.parse_args()

SRC_DIR = Path(__file__).parent.parent.parent #this file is in YOLO/src/valid/native_latency, 3 parents up is YOLO/src

#NOTE: GOAL is to run both yoloNATIVE.py which captures yolo's native benchmark latency with a single compiled graph for inference steps
#      and YOLO/src/benchmark/main.py which captures harness' benchmark latency with a single compiled graph for inference steps
#      capture the latency of preprocess, inference, and postprocess cli outputs from both, and validate they are all within 5%

#CAPTURE NATIVE TIMING
native_times = {
            "preprocess": [],
            "inference": [],
            "postprocess": []
}
for i in range(args.iterations):
    print(f"native run {i + 1} of {args.iterations}")
    native_result = subprocess.run([sys.executable, "yolo/yoloNATIVE.py", "--step", str(args.step)],
                    cwd = SRC_DIR, capture_output = True, text = True) #run from YOLO/src, text = True gives strings instead of bytes
    #prints lines like preprocess 1.42
    for line in native_result.stdout.splitlines():
        stage = line.split()
        if stage and stage[0] in native_times: #line starts with a stage name
            native_times[stage[0]].append(float(stage[-1])) #last entry on line should be time


#CAPTURE HARNESS TIMING
harness_names = {"pre_processing": "preprocess", "inference": "inference", "post_processing": "postprocess"}
harness_times = {
            "preprocess": [],
            "inference": [],
            "postprocess": []
} 
for i in range(args.iterations):
    print(f"harness run {i + 1} of {args.iterations}")
    harness_result = subprocess.run([sys.executable, "unified_compiled/main.py", "--setting", "profile", "--step", str(args.step)],
                            cwd = SRC_DIR, capture_output = True, text = True) #run from YOLO/src, text = True gives strings instead of bytes
    #harness prints lines like "  pre_processing   TOTAL WALL TIME   1.46"
for line in harness_result.stdout.splitlines():
    stage = line.split()
    if stage and stage[0] in harness_names and "TOTAL" in line: #only the PASS 1 total lines
        harness_times[harness_names[stage[0]]].append(float(line.split("(")[1].split()[0]) if "(" in line else float(stage[-1])) #per image time (inside the parentheses), ultralytics' speeds are per image too

#ERROR HANDLING OF SUBPROCESSES

if len(native_times["inference"]) != args.iterations:
    print(native_result.stderr)
    sys.exit("ERROR: yoloNATIVE.py DIDN'T PRINT TIMES")
if len(harness_times["inference"]) != args.iterations:
    print(harness_result.stderr)
    sys.exit("ERROR: unified_compiled/main.py DIDN'T PRINT TIMES")

#capture average mean difference in each stage for optional validation check (NOTE: no stage +/-5% diff)
native_med = {
            "preprocess": 0,
            "inference":0,
            "postprocess": 0
}
harness_med = {
            "preprocess": 0,
            "inference": 0,
            "postprocess": 0
}
native_total = 0; harness_total = 0;
print(f"\n  {'STAGE':<12} {'native ms':>10} {'harness ms':>11} {'diff':>8}")
for stage in native_times:
    native_med[stage] += sum(native_times[stage]) / len(native_times[stage]); native_total += native_med[stage]
    harness_med[stage] += sum(harness_times[stage]) / len(harness_times[stage]); harness_total += harness_med[stage]
    print(f"  {stage:<12} {native_med[stage]:>10.3f} {harness_med[stage]:>11.3f} {(harness_med[stage] - native_med[stage]) / native_med[stage] * 100:>+7.1f}%")
#convert differences to percentage

total_diff = ( (harness_total-native_total)/ native_total) * 100
print(f"  {'TOTAL':<12} {native_total:>10.3f} {harness_total:>11.3f} {total_diff:>+7.1f}%")
if abs(total_diff) > args.accuracy:
    print(f"    ===Holistic Native latency validation check FAILED!\n    ===Expected no greater than {args.accuracy}% overall latency difference, actual: {total_diff:>+.1f}%")
else:
    print(f"    ===Holistic Native latency validation check PASSED!\n    ===Expected no greater than {args.accuracy}% overall latency difference, actual: {total_diff:>+.1f}%")

pre_diff = ( (harness_med["preprocess"]-native_med["preprocess"])/ native_med["preprocess"]) * 100 
#print(f"  {'TOTAL':<12} {native_total:>10.3f} {harness_total:>11.3f} {pre_diff:>+7.1f}%")
if abs(pre_diff) > args.accuracy:
    print(f"    ===Optional Substage Native latency validation check FAILED!\n    ===Expected no greater than {args.accuracy}% preprocess latency difference, actual: {pre_diff:>+.1f}%")
else:
    print(f"    ===Optional Substage Native latency validation check PASSED!\n    ===Expected no greater than {args.accuracy}% preprocess latency difference, actual: {pre_diff:>+.1f}%")
inf_diff = ( (harness_med["inference"]-native_med["inference"])/ native_med["inference"]) * 100
#print(f"  {'TOTAL':<12} {native_total:>10.3f} {harness_total:>11.3f} {inf_diff:>+7.1f}%")
if abs(inf_diff) > args.accuracy:
    print(f"    ===Optional Substage Native latency validation check FAILED!\n    ===Expected no greater than {args.accuracy}% inference latency difference, actual: {inf_diff:>+.1f}%")
else:
    print(f"    ===Optional Substage Native latency validation check PASSED!\n    ===Expected no greater than {args.accuracy}% inference latency difference, actual: {inf_diff:>+.1f}%")
post_diff = ( (harness_med["postprocess"]-native_med["postprocess"])/ native_med["postprocess"]) * 100
#print(f"  {'TOTAL':<12} {native_total:>10.3f} {harness_total:>11.3f} {post_diff:>+7.1f}%")
if abs(post_diff) > args.accuracy:
    print(f"    ===Optional Substage Native latency validation check FAILED!\n    ===Expected no greater than {args.accuracy}% postprocess latency difference, actual: {post_diff:>+.1f}%")
else:
    print(f"    ===Optional Substage Native latency validation check PASSED!\n    ===Expected no greater than {args.accuracy}% postprocess latency difference, actual: {post_diff:>+.1f}%")
