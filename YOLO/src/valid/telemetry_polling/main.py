import sys
from pathlib import Path
import time
import argparse
import subprocess #execute fresh hardware run of harness
import statistics #mean of the time between readings

#NOTE: GOAL is to show the tt-smi sampler (TT_SAMPLER in xla_yolo/hardware.py) really writes a reading every 100 ms
#      while the card is working, the tt equivalent of tegrastats --interval 100 on the jetson
#      1. fresh hardware run (xla_yolo/main.py --setting hardware), full pipeline and stage windows
#      2. pick out every tt_smi.csv reading inside each window that run stored (hw_metrics.csv / hw_stages.csv rows)
#      3. time between back to back readings: the average must be within --accuracy % of 100 ms
#         and no single gap may be over 200 ms (a gap that long means at least one reading was missed)
#      the sampler sleeps 100 ms after reading the card, so slow reads add to every interval and show up in the average
#USAGE: python main.py --step 4

parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1)") #used to create accurate subset of db
parser.add_argument("--accuracy", type = int, choices = [1,2,3,4,5,6,7,8,9,10], default = 5, help = "max difference of the average polling interval from 100 ms as a %% (defaulted to 5%%)")
args = parser.parse_args()

SRC_DIR = Path(__file__).parent.parent.parent #this file is in YOLO/src/valid/telemetry_polling, 3 parents up is YOLO/src
BENCHMARK_DIR = SRC_DIR.parent / "benchmarking" #one level up from src is YOLO
INTERVAL = 100 #ms, NOTE: must match the interval identify_monitor() in xla_yolo/hardware.py passes to the sampler
MAX_GAP = 2 * INTERVAL #ms, a gap this long means at least one reading was missed

#FRESH RUN
print("fresh hardware run (takes a few minutes)")
fresh_start = time.time() #record timestamp of fresh run to locate start of its recorded readings and windows
result = subprocess.run([sys.executable, "xla_yolo/main.py", "--setting", "hardware", "--step", str(args.step)],
                        cwd = SRC_DIR, capture_output = True, text = True) #run from YOLO/src, text = True gives strings instead of bytes
if result.returncode != 0:
    print(result.stdout[-3000:]); print(result.stderr[-3000:]) #end of the run's output, where the error is
    sys.exit("ERROR: FRESH HARDWARE RUN FAILED (see the output above)")

#time stamp of every tt_smi.csv reading the fresh run wrote
stamps = []
for line in open(BENCHMARK_DIR / "tt_smi.csv"):
    first = line.split(",", 1)[0]
    try:
        t = float(first)
    except ValueError: #if line doesn't have a timestamp (the csv header)
        continue  #skip current line
    if t >= fresh_start:
        stamps.append(t)

#windows the fresh run stored: full pipeline (hw_metrics.csv, 7 columns) and every stage (hw_stages.csv, 8 columns)
windows = [] #(name, start, end)
for path, n_columns in ((BENCHMARK_DIR / "hw_metrics.csv", 7), (BENCHMARK_DIR / "hw_stages.csv", 8)):
    for line in open(path):
        row = line.strip().split(",")
        if len(row) == n_columns and row[0] == "tt_smi" and row[2].isdigit() and float(row[3]) >= fresh_start:
            windows.append((row[1], float(row[3]), float(row[4])))
if not windows:
    sys.exit("ERROR: FRESH RUN WROTE NO TT_SMI ROWS TO hw_metrics.csv / hw_stages.csv")

#DISPLAY RESULTS
print(f"\n  tt-smi readings inside each window of the fresh run (step {args.step}), expected one every {INTERVAL} ms")
print(f"  {'WINDOW':<16} {'seconds':>8} {'readings':>9} {'expected':>9} {'avg gap[ms]':>12} {'max gap[ms]':>12}")
failed = [] #stores windows that failed validation criteria
for name, start, end in windows:
    inside = [t for t in stamps if start <= t <= end]
    gaps = [(b - a) * 1000 for a, b in zip(inside, inside[1:])] #ms between back to back readings
    expected = (end - start) * 1000 / INTERVAL #readings a perfect 100 ms sampler would write in the window
    if not gaps: #need at least 2 readings for a gap
        print(f"  {name:<16} {end - start:>8.1f} {len(inside):>9} {expected:>9.0f} {'-':>12} {'-':>12}")
        failed.append(f"{name} (fewer than 2 readings)")
        continue
    avg = statistics.mean(gaps)
    print(f"  {name:<16} {end - start:>8.1f} {len(inside):>9} {expected:>9.0f} {avg:>12.1f} {max(gaps):>12.1f}")
    if abs(avg - INTERVAL) > INTERVAL * args.accuracy / 100 or max(gaps) > MAX_GAP:
        failed.append(name)

#VALIDATE RESULTS
if failed:
    print(f"\n    ===Telemetry polling validation check FAILED!\n    ===Expected an average gap within {args.accuracy}% of {INTERVAL} ms and no gap over {MAX_GAP} ms in every window, failed: {', '.join(failed)}")
else:
    print(f"\n    ===Telemetry polling validation check PASSED!\n    ===Every window's average gap is within {args.accuracy}% of {INTERVAL} ms with no gap over {MAX_GAP} ms")
