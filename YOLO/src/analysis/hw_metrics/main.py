import sys
import re
from pathlib import Path
import time
import argparse #cli args
import subprocess #execute fresh hardware run of harness

from helper import report

#prints the average and max power, temperature and GPU load from tegrastats.log for each hardware run
#USAGE: python3 hw_simple.py [benchmarking folder]
#needs: tegrastats.log (lines start with the time.time() stamp), hw_metrics.csv, and hw_stages.csv files
#shows: the latest full pipeline run for each batch size, then the latest idle / pre / inference / post stage runs
parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64], default = None, help = "only show this batch size (with --fresh: the batch size to run, defaulted to 1)")
#NOTE: batch size doesnt default to 1 so script doesnt default to displaying only logs with batch size of 1
parser.add_argument("--all", action = "store_true", help = "show every recorded run, not just the latest or indicated batch_size of interest")
parser.add_argument("--fresh", action = "store_true", help = "run a fresh harness with hardware setting, and show only that run")
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "with --fresh: every nth image of the database (defaulted to 1)")
args = parser.parse_args()

SRC_DIR = Path(__file__).parent.parent.parent #this file is in YOLO/src/analysis/hw_metrics, 3 parents up is YOLO/src
BENCHMARK_DIR = SRC_DIR.parent / "benchmarking" #one level up from src is YOLO

#FRESH RUN: indicated by --fresh tag
fresh_start = 0.0 #record timestamp of fresh run to locate start of its recorded metrics in logs and csvs
if args.fresh:
    batch = args.batch_size or 1
    elapsed_time = time.time()
    cmd = [sys.executable, "substage_compiled/main.py", "--setting", "hardware", "--batch_size", str(batch), "--step", str(args.step)]
    print("  $ " + " ".join(cmd)) #dispaly exact command to allow for it to be rerun by hand
    if subprocess.run(cmd, cwd = SRC_DIR).returncode != 0:
        sys.exit("ERROR: FRESH HARDWARE RUN FAILED (see the output above)")
    args.batch_size = batch
    
#read the log: keep (time, line) for every line that starts with a time stamp
readings = []
for line in open(BENCHMARK_DIR / "tegrastats.log"):
    first = line.split(" ", 1)[0]
    try:
        readings.append((float(first), line))
    except ValueError: #if line doesn't have a timestamp
        pass  #skip current line

#full pipeline runs
#NOTE: hw_metrics.csv rows are monitor,phase,batch_size,start,end
runs = [] #(batch size, start, end) in file order (oldest first)
for line in open(BENCHMARK_DIR / "hw_metrics.csv"):
    row = line.strip().split(",")
    if len(row) == 5 and row[0] == "tegrastats" and row[2].isdigit() and float(row[3]) >= fresh_start:
        if args.batch_size is None or int(row[2]) == args.batch_size:
            runs.append((int(row[2]), float(row[3]), float(row[4])))
if not args.all: #keep only the latest run of each batch size
    latest = {b: (b, s, e) for b, s, e in runs} #later rows replace earlier ones
    runs = [latest[b] for b in sorted(latest)]
for b, start, end in runs:
    report(f"full pipeline, batch {b}", start, end, readings)

#stage average performance from 5 seconds of runtime
#NOTE: hw_stages.csv rows are monitor,stage,(batch_size,)start,end,iterations
stages_path = BENCHMARK_DIR / "hw_stages.csv"
if stages_path.exists():
    stage_runs = [] #(stage, start, end, iterations) in file order (oldest first)
    for line in open(stages_path):
        row = line.strip().split(",")
        if row[0] == "tegrastats" and len(row) in (5, 6) and row[-1].isdigit() and float(row[-3]) >= fresh_start:
            stage_runs.append((row[1], float(row[-3]), float(row[-2]), int(row[-1])))
    if not args.all: #keep only the latest run of each stage
        latest = {name: (name, s, e, n) for name, s, e, n in stage_runs}
        stage_runs = [latest[name] for name in ("idle", "pre_processing", "inference", "post_processing") if name in latest]
    for name, start, end, n in stage_runs:
        report(f"stage {name} (batch 1, {(end - start) * 1000 / n:.2f} ms per run)", start, end, readings)

if not runs and args.fresh:
    print("ERROR: fresh run wrote no tegrastats rows verify with harness output above")
