import sys
from pathlib import Path
import time
import argparse #cli args
import subprocess #execute fresh hardware run of harness

from helper import report, wanted

#prints the average and max power, temperature and GPU load from tegrastats.log for each hardware run
#USAGE: python3 hw_summary.py                                     latest run of every batch size / step / optimization + latest stage runs
#       python3 hw_summary.py --batch_size 8 --optimization eager only runs that match every option given
#       python3 hw_summary.py --all                               every recorded run, oldest first
#       python3 hw_summary.py --fresh --batch_size 8 --step 4     runs the harness hardware setting now, then shows only that run

parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64], default = None, help = "only show this batch size (with --fresh: the batch size to run, defaulted to 1)")
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = None, help = "only show this step (with --fresh: every nth image of the database to run, defaulted to 1)")
parser.add_argument("--optimization", choices = ["inductor", "aot_eager", "eager"], default = None, help = "only show this torch.compile backend (with --fresh: the backend to run, defaulted to inductor)")
parser.add_argument("--all", action = "store_true", help = "show every recorded run, not just the latest of each batch size / step / optimization")
parser.add_argument("--fresh", action = "store_true", help = "run a fresh harness with hardware setting, and show only that run")
args = parser.parse_args()
#NOTE: batch size, step and optimization don't default to anything, so the script shows every stored run unless one is given

SRC_DIR = Path(__file__).parent.parent.parent #this file is in YOLO/src/analysis/hw_metrics, 3 parents up is YOLO/src
BENCHMARK_DIR = SRC_DIR.parent / "benchmarking" #one level up from src is YOLO

#FRESH RUN: indicated by --fresh tag
fresh_start = 0.0 #record timestamp of fresh run to locate start of its recorded metrics in logs and csvs
if args.fresh:
    args.batch_size = args.batch_size or 1
    args.step = args.step or 1
    args.optimization = args.optimization or "inductor"
    fresh_start = time.time()
    cmd = [sys.executable, "substages_compiled/main.py", "--setting", "hardware", "--batch_size", str(args.batch_size), "--step", str(args.step),
           "--optimization", args.optimization]
    print("  $ " + " ".join(cmd)) #display exact command to allow for it to be rerun by hand
    if subprocess.run(cmd, cwd = SRC_DIR).returncode != 0:
        sys.exit("ERROR: FRESH HARDWARE RUN FAILED (see the output above)")
    
#read the log: keep (time, line) for every line that starts with a time stamp
readings = []
for line in open(BENCHMARK_DIR / "tegrastats.log"):
    first = line.split(" ", 1)[0]
    try:
        readings.append((float(first), line))
    except ValueError: #if line doesn't have a timestamp
        pass  #skip current line

#full pipeline runs
#NOTE: hw_metrics.csv rows are monitor,phase,batch_size,start,end,step,optimization
runs = [] #(batch size, step, optimization, start, end) in file order (oldest first)
for line in open(BENCHMARK_DIR / "hw_metrics.csv"):
    row = line.strip().split(",")
    if len(row) == 7 and row[0] == "tegrastats" and row[2].isdigit() and float(row[3]) >= fresh_start:
        b, step, opt = int(row[2]), int(row[5]), row[6]
        if wanted(b, step, opt, args):
            runs.append((b, step, opt, float(row[3]), float(row[4])))
if not args.all: #keep only the latest run of each batch size / step / optimization
    latest = {r[:3]: r for r in runs} #later rows replace earlier ones
    runs = [latest[k] for k in sorted(latest)]
for b, step, opt, start, end in runs:
    report(f"full pipeline, batch {b}, step {step}, {opt}", start, end, readings)

#stage average performance from 5 seconds of runtime
#NOTE: hw_stages.csv rows are monitor,stage,batch_size,start,end,iterations,step,optimization
stages_path = BENCHMARK_DIR / "hw_stages.csv"
stage_runs = [] #(stage, step, optimization, start, end, iterations) in file order (oldest first)
if stages_path.exists():
    for line in open(stages_path):
        row = line.strip().split(",")
        if len(row) == 8 and row[0] == "tegrastats" and row[5].isdigit() and float(row[3]) >= fresh_start:
            step, opt = int(row[6]), row[7]
            if wanted(None, step, opt, args):
                stage_runs.append((row[1], step, opt, float(row[3]), float(row[4]), int(row[5])))
    if not args.all: #keep only the latest run of each stage / step / optimization
        order = ("idle", "pre_processing", "inference", "post_processing")
        latest = {(r[1], r[2], r[0]): r for r in stage_runs}
        stage_runs = [latest[k] for k in sorted(latest, key = lambda k: (k[0], k[1], order.index(k[2]) if k[2] in order else 9))]
    for name, step, opt, start, end, n in stage_runs:
        report(f"stage {name} (batch 1, step {step}, {opt}, {(end - start) * 1000 / n:.2f} ms per run)", start, end, readings)

if not runs and args.fresh:
    print("ERROR: FRESH RUN WROTE NO TEGRASTATS ROWS IN LOGS")
