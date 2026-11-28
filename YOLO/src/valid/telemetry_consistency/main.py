import sys
from pathlib import Path
import argparse
import statistics #mean / std dev across runs

from helper import fresh_run, below_idle, STAGES

#NOTE: GOAL is a fresh hardware run (analysis/hw_metrics/main.py --fresh), then checking that every stage
#      (pre_processing / inference / post_processing) has power, current and temperatures at least as high as idle
#      temperatures get --temp_tolerance: the sensors read in ~0.1-0.5 C steps and a short stage only heats the chip a little,
#      so a stage that reads a hair under idle is sensor noise, not a cooler chip (the harness cools down before every stage,
#      config.COOLDOWN, so every stage starts from the same resting temperature)
#      AICLK and VCORE are shown but not checked: the card's firmware sets the clock (and the voltage that goes with it) to busy
#      as soon as the harness opens the card, so they don't follow the work (idle included)
#      tt-smi only reads the card (no host cpu rails like tegrastats), so the mostly host pre_processing / post_processing read close to idle
#USAGE: python main.py --step 4 --batch_size 16 --optimization 1 --iterations 3

parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1)") #used to create accurate subset of db
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64], default = 1, help = "batch size of the full pipeline and stage hardware runs (defaulted to 1)")
parser.add_argument("--optimization", type = int, choices = [0, 1, 2], default = 0, help = "determines level of TT-XLA compiler optimization (defaulted to 0)")
parser.add_argument("--iterations", type = int, choices = [1,2,3,4,5,6,7,8,9,10], default = 1, help = "how many fresh hardware runs to check one after another (defaulted to 1, a few min each)")
parser.add_argument("--temp_tolerance", type = float, default = 0.5, help = "degrees C a stage's average temperature may read below idle and still pass (defaulted to 0.5, 0 = strict)")
args = parser.parse_args()

SRC_DIR = Path(__file__).parent.parent.parent #this file is in YOLO/src/valid/telemetry_consistency, 3 parents up is YOLO/src

#RUN AND CHECK EVERY ITERATION
all_runs, failed = [], [] #store data from runs, and failed data separately
for i in range(args.iterations):
    print(f"fresh hardware run {i + 1} of {args.iterations} (takes a few minutes)")
    metrics = fresh_run(SRC_DIR, args)
    all_runs.append(metrics)
    bad = below_idle(metrics, args)
    print(f"  run {i + 1}: {'PASSED' if not bad else 'below idle: ' + ', '.join(bad)}")
    failed += [f"run {i + 1} {b}" for b in bad]

#DISPLAY mean of every metric over the runs (std dev across runs when there is more than one)
print(f"\n  batch {args.batch_size}, step {args.step}, opt {args.optimization}: mean over {args.iterations} run(s){' (+/- std dev across runs)' if args.iterations > 1 else ''}")
print(f"  {'METRIC':<20} {'idle':>16}" + "".join(f" {s:>18}" for s in STAGES))
for key in all_runs[0]["idle"]:
    row = f"  {key:<20}"
    for s in ("idle",) + STAGES:
        v = [m[s][key] for m in all_runs if key in m[s]]
        if not v:
            row += f" {'-':>{16 if s == 'idle' else 18}}"
            continue
        cell = f"{statistics.mean(v):.2f}" + (f" +/-{statistics.stdev(v):.2f}" if len(v) > 1 else "")
        row += f" {cell:>{16 if s == 'idle' else 18}}"
    print(row)
#NOTE: every power reading, current and temperature of every stage must be >= idle in every run (temperatures within --temp_tolerance)
if failed: #if failed run exists
    print(f"    ===Telemetry consistency validation check FAILED!\n    ===Expected every stage's power, current and temperature (within {args.temp_tolerance} C) >= idle in all {args.iterations} run(s), below idle: {', '.join(failed)}")
else:
    print(f"    ===Telemetry consistency validation check PASSED!\n    ===Every stage's power, current and temperature (within {args.temp_tolerance} C) >= idle in all {args.iterations} run(s)")
