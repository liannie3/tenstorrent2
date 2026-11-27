import sys
import subprocess
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--iterations", type = int, choices = [1,2,3,4,5,6,7,8,9,10],  default = 5, help = "determines how many iterations are run (defaulted to 5)") #used to allow user tuning of noise reducton
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1)") #used to create accurate subset of db
parser.add_argument("--accuracy", type = int, choices = [1,2,3,4,5,6,7,8,9,10],  default = 5, help = "determines the degree accuracy to validate against (defaulted to 5%%)") #used to allow user tuning accuracy threshold
parser.add_argument("--setting", choices = ["sub_events", "stage", "e2e", "sub_profiled"], default = "stage", help = "determines the type of harness run to validate (defaulted to stage)") #NOTE: profiled should use the profiled stage share latency times
parser.add_argument("--optimization", choices = ["inductor", "aot_eager", "eager"], default = "inductor", help = "determines torch.compile backend, aot_eager = no fusion / no Triton optimization (defaulted to inductor)") #set level of compile time optimization via compilation backend tuning
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64] , default = 1, help = "determines throughput of images sent through the model per batch (defaulted to 1)")

args = parser.parse_args()

SRC_DIR = Path(__file__).parent.parent.parent #this file is in YOLO/src/valid/run_repeat, 3 parents up is YOLO/src

run_data = {i: for i in range(args.iterations) } #create dict to store up to iteration different runs

for i in range(args.iterations):
    print(f"{args.setting} run {i + 1} of {args.iterations}")

    #CAPTURE RESULTS
    if args.setting == "e2e": #end to end latency run repeat verification
        result = subprocess.run([sys.executable, "substages_compiled/main.py", "--step", str(args.step), "--setting", "e2e", "--otpimization", args.optimization,
                                 "batch_size", args.batch_size], cwd = SRC_DIR, capture_output = True, text = True) #run from YOLO/src, text = True gives strings instead of bytes
        for line in result.stdout.splitlines():
            stage = line.split()
            if stage[:2] == ["BATCH", "MEDIAN"]: #identified line that prints batch median
                e2e_times.append(float(stage[-1])) #last entry on line should be time

    elif args.setting == "sub_events": #substep timing with cuda gpu events run repeat verification
        result = subprocess.run([sys.executable, "substages_compiled/main.py", "--step", str(args.step), "--setting", "latency", "--otpimization", args.optimization,
                                 "batch_size", args.batch_size], cwd = SRC_DIR, capture_output = True, text = True) #run from YOLO/src, text = True gives strings instead of bytes
        harness_names = {"pre_processing": "preprocess", "inference": "inference", "post_processing": "postprocess"}
        for line in result.stdout.splitlines():
            stage = line.split()
            if len(stage) >= 3 and stage[0] in harness_names:
                if "TOTAL WALL TIME" in line:
                    wall_times[harness_names[stage[0]]].append(float(stage[-1]))
                elif "SUM OF STEPS" not in line:
                    sub_times.setdefault(stage[1], []).append(float(stage[-1])) #e.g. "A_LetterBox"

    elif args.setting == "stage": #substep timing with perf counters and syncs in ULTRALYTICS style
        result = subprocess.run([sys.executable, "substages_compiled/main.py", "--step", str(args.step), "--setting", "latency", "--otpimization", args.optimization,
                                         "batch_size", args.batch_size], cwd = SRC_DIR, capture_output = True, text = True) #run from YOLO/src, text = True gives strings instead of bytes
            harness_names = {"pre_processing": "preprocess", "inference": "inference", "post_processing": "postprocess"}
            for line in result.stdout.splitlines():
                stage = line.split()
                if len(stage) >= 3 and stage[0] in harness_names:
                    if "TOTAL WALL TIME" in line:
                        wall_times[harness_names[stage[0]]].append(float(stage[-1]))
                    elif "SUM OF STEPS" not in line:
                        sub_times.setdefault(stage[1], []).append(float(stage[-1])) #e.g. "A_LetterBox"

    elif args.setting == "sub_profiled":
        result = subprocess.run([sys.executable, "analysis/stage_share_latency/stage_share.py/main.py", "--step", str(args.step),
                                 "batch_size", args.batch_size], cwd = SRC_DIR, capture_output = True, text = True) #run from YOLO/src, text = True gives strings instead of bytes
        for line in result.stdout.splitlines():
            stage = line.split()
            if "EST. TIME PER IMAGE" in line:
                share_est[stage[0]].append(float(stage[5])) #ms per image of the stage
            elif len(stage) == 7 and stage[0] in ("preprocess", "inference", "postprocess"):
                share_pct.setdefault(stage[1], []).append(float(stage[5])) #avg % of the subste




