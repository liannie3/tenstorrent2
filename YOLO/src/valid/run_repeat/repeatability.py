import sys
import subprocess
from pathlib import Path
import argparse
import statistics #easily calculate std and mean

parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--iterations", type = int, choices = [1,2,3,4,5,6,7,8,9,10],  default = 5, help = "determines how many iterations are run (defaulted to 5)") #used to allow user tuning of noise reducton
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1)") #used to create accurate subset of db
parser.add_argument("--accuracy", type = int, choices = [1,2,3,4,5,6,7,8,9,10],  default = 5, help = "determines the degree accuracy to validate against (defaulted to 5%%)") #used to allow user tuning accuracy threshold
parser.add_argument("--setting", choices = ["sub_events", "stage", "e2e"], default = "stage", help = "determines the type of harness run to validate (defaulted to stage)") #NOTE: no profiled stage share setting, tracy's overhead makes repeated profiled runs too slow
parser.add_argument("--optimization", type = int, choices = [0, 1, 2], default = 0, help = "determines level of TT-XLA compiler optimization (defaulted to 0)")
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64] , default = 1, help = "determines throughput of images sent through the model per batch (defaulted to 1)")
args = parser.parse_args()

SRC_DIR = Path(__file__).parent.parent.parent #this file is in YOLO/src/valid/run_repeat, 3 parents up is YOLO/src

harness_args = ["--step", str(args.step), "--optimization", str(args.optimization), "--batch_size", str(args.batch_size) ]
MIN_ABS = 0.05
times = {} #create dict to store up to iteration different runs
def store(key, value): #store a value inside of times dict
    times.setdefault(key, []).append(value)

for i in range(args.iterations):
    print(f"{args.setting} run {i + 1} of {args.iterations}")

    #CAPTURE RESULTS
    if args.setting == "e2e": #end to end latency run repeat verification
        result = subprocess.run([sys.executable, "xla_yolo/main.py", "--setting", "e2e"] + harness_args,
                                 cwd = SRC_DIR, capture_output = True, text = True) #run from YOLO/src, text = True gives strings instead of bytes
        for line in result.stdout.splitlines():
            stage = line.split()
            if stage[:2] == ["BATCH", "MEDIAN"]: #identified line that prints batch median
                store("e2e BATCH MEDIAN", float(line.split("(")[0].split()[-1])) #batch time, the number before the "(... ms per image)" note

    elif args.setting == "sub_events" or args.setting == "stage": #ULTRALYITICS-style sync substage time abd substep timing with card syncs run repeat verification and
                                                                  #NOTE: both are recorded in latency run of harness PASS 1 and PASS 2 respectively
                                                                  #NOTE: sub_events keeps the CUDA setting name, PASS 2 syncs the card before and after every card substep (PyTorch/XLA has no gpu events, see timing.py)
        result = subprocess.run([sys.executable, "xla_yolo/main.py", "--setting", "latency"] + harness_args,
                                 cwd = SRC_DIR, capture_output = True, text = True) #run from YOLO/src, text = True gives strings instead of bytes
        HARNESS_NAMES = {"pre_processing": "preprocess", "inference": "inference", "post_processing": "postprocess"}
        for line in result.stdout.splitlines():
            stage = line.split()
            if len(stage) >= 3 and stage[0] in HARNESS_NAMES:
                if "TOTAL WALL TIME" in line and args.setting == "stage":
                    store(HARNESS_NAMES[stage[0]], float(line.split("(")[0].split()[-1])) #batch time, the number before the "(... ms per image)" note
                elif "TOTAL WALL TIME" not in line and "SUM OF STEPS" not in line and args.setting == "sub_events":
                    store(f"{HARNESS_NAMES[stage[0]]} {stage[1]}", float(line.split("(")[0].split()[-1])) #e.g. "preprocess A_LetterBox", batch time before the "(... ms per image)" note

    #ERROR HANDLING OF SUBPROCESES
    if result.returncode != 0 or not times or any(len(v) != i + 1 for v in times.values()):
        print(result.stdout[-3000:]); print(result.stderr[-3000:]) #end of the run's output, where the error is
        sys.exit(f"ERROR: RUN {i + 1} FAILED OR DIDN'T PRINT EXPECTED LATENCIES")

#NOTE: recall validation requirement for test is:
#      Standard deviation of end-to-end latency of any run through each subsystem should be within 5% (or args.accuracy) of the mean latency.
#DISPLAY RESULTS
print(f"\n  {'NAME':<34} {'mean[ms]':>9} {'std dev[min]':>9} {'min[ms]':>9} {'max[ms]':>9} {'std/mean[%]':>9}")
failed = [] #stores runs that failed validation criteria
for key, values in times.items(): #loop through iterations stored times
    mean = statistics.mean(values)
    std = statistics.stdev(values) #sample standard deviation (n - 1)
    cv = ( (std / mean) * 100) if mean else 0.0 #compute coefficient of variation, std dev as % of the mean
    print(f"  {key:<34} {mean:>9.3f} {std:>9.3f} {min(values):>9.3f} {max(values):>9.3f} {cv:>8.1f}%")
    if cv > args.accuracy and std > MIN_ABS:
        failed.append(key)

#VALIDATE RESULTS
if failed:
    print(f"\n    ===Run repeatability validation check FAILED!\n    ===Expected standard deviation within {args.accuracy}% of the mean across {args.iterations} runs,\n    failed: {', '.join(failed)}")
else:
    print(f"\n    ===Run repeatability validation check PASSED!\n    ===Standard deviation within {args.accuracy}% of the mean for every latency across {args.iterations} runs")
