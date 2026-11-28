import sys
import csv
import time
import argparse
import subprocess #runs the harness / nsys
from pathlib import Path

#NOTE: GOAL is to fill benchmarking/ with every run the graphs notebook needs, for every combination of the lists below
#      runs ONE AT A TIME (nothing shares the GPU or a tegrastats window), and skips any combination that's already
#      in the logs, so rerunning this only fills the gaps and never adds the same run twice
#      order: latency + e2e first (fast), then nsys + stage share, then hardware (slow, ~4 min each), so the quick graphs work early
#NOTE: hardware rows don't store precision, so hardware only runs with the FIRST precision in PRECISIONS
#      (a second precision would be indistinguishable from the first in hw_metrics.csv / hw_stages.csv)
#NOTE: nsys profiles are named nsys_e2e_b<batch>_step<step>_<optimization>_<precision> so different settings never overwrite each other
#      (the stage share script's nsys_e2e_b<batch> files are left alone)
#NOTE: stage share only takes --batch_size and --step, so the harness runs it with its defaults (inductor, bf16)
#      it only runs for that optimization + precision, any other combination is left out
#LOCATION: YOLO/src/analysis/graphs/fill_logs.py (next to the graphs notebook)
#USAGE: python fill_logs.py --dry_run     list what would run, run nothing
#       python fill_logs.py               run everything that's missing (leave it in tmux)

#WHAT TO RUN (edit these)
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
STEPS = [4]
OPTIMIZATIONS = ["inductor"]
PRECISIONS = ["bf16"]
SETTINGS = ["latency", "e2e", "nsys", "stage_share", "hardware"] #remove any you don't want

parser = argparse.ArgumentParser()
parser.add_argument("--dry_run", action = "store_true", help = "list the runs that are missing, run nothing")
args = parser.parse_args()

SRC_DIR = Path(__file__).parent.parent.parent #this file is in YOLO/src/analysis/graphs, 3 parents up is YOLO/src
BENCH_DIR = SRC_DIR.parent / "benchmarking"
HARNESS = "substage_compiled/main.py" #NOTE: must match the harness folder name
STAGE_SHARE = "analysis/stage_share_latency/main.py"
LOG_DIR = BENCH_DIR / "fill_logs" #full output of every run, for when one fails
LOG_DIR.mkdir(parents = True, exist_ok = True)

#helper func that reads a csv into a list of dicts (empty list if it doesn't exist yet)
def read(name):
    path = BENCH_DIR / name
    if not path.exists():
        return []
    return list(csv.DictReader(open(path)))

#helper func that is True when a run with these settings is already in the logs
def done(setting, b, step, opt, prec):
    if setting == "stage_share":
        for r in read("stage_share.csv"):
            if r.get("batch_size") == str(b) and r.get("step") == str(step):
                return True
        return False
    if setting == "nsys":
        return (BENCH_DIR / f"nsys_e2e_b{b}_step{step}_{opt}_{prec}.sqlite").exists()
    if setting == "hardware": #no precision column in the hardware csvs
        for r in read("hw_stages.csv"):
            if r.get("monitor") == "tegrastats" and r.get("batch_size") == str(b) and r.get("step") == str(step) and r.get("optimization") == opt:
                return True
        return False
    wanted_pass = "substep_timing" if setting == "latency" else "e2e" #latency writes PASS 1 then PASS 2, PASS 2 last means it finished
    for r in read("stage_timings.csv"):
        if (r.get("pass") == wanted_pass and r.get("batch_size") == str(b) and r.get("step") == str(step)
                and r.get("optimization") == opt and r.get("precision") == prec):
            return True
    return False

#helper func that runs one command from YOLO/src, saves its whole output to a log, and returns True if it worked
def run(cmd, log_name):
    log = LOG_DIR / f"{log_name}.log"
    start = time.time()
    try:
        result = subprocess.run([str(c) for c in cmd], cwd = SRC_DIR, capture_output = True, text = True)
    except FileNotFoundError: #program not installed (e.g. nsys), count it as failed and keep going
        log.write_text("$ " + " ".join(str(c) for c in cmd) + f"\n\nERROR: {cmd[0]} NOT FOUND")
        print(f"    FAILED: {cmd[0]} not found")
        return False
    log.write_text("$ " + " ".join(str(c) for c in cmd) + "\n\n" + result.stdout + "\n" + result.stderr)
    print(f"    {'done' if result.returncode == 0 else 'FAILED'} in {time.time() - start:.0f} s  (output: {log})")
    return result.returncode == 0

#every missing run, in order
todo = []
for setting in SETTINGS:
    for prec in PRECISIONS:
        if setting == "hardware" and prec != PRECISIONS[0]:
            continue #see NOTE at the top
        if setting == "stage_share" and prec != "bf16":
            continue #see NOTE at the top
        for opt in OPTIMIZATIONS:
            if setting == "stage_share" and opt != "inductor":
                continue
            for step in STEPS:
                for b in BATCH_SIZES:
                    if not done(setting, b, step, opt, prec):
                        todo.append((setting, b, step, opt, prec))

print(f"{len(todo)} runs missing")
for setting, b, step, opt, prec in todo:
    print(f"  {setting:<11} batch {b:<3} step {step:<3} {opt:<10} {prec}")
if args.dry_run or not todo:
    sys.exit()

results = [] #(name, "done" / "FAILED" / "skipped") for the summary at the end
start_all = time.time()
for i, (setting, b, step, opt, prec) in enumerate(todo):
    name = f"{setting}_b{b}_step{step}_{opt}_{prec}"
    print(f"\n[{i + 1}/{len(todo)}] {name}  ({time.strftime('%H:%M:%S')})")
    if done(setting, b, step, opt, prec): #another run filled it in the meantime
        print("    already in the logs, skipped")
        results.append((name, "skipped"))
        continue
    harness_args = ["--batch_size", b, "--step", step, "--optimization", opt, "--precision", prec]
    if setting == "nsys":
        profile = BENCH_DIR / f"nsys_e2e_b{b}_step{step}_{opt}_{prec}"
        ok = run(["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true", "-o", profile,
                  sys.executable, HARNESS, "--setting", "e2e", "--profiled"] + harness_args, name)
        if ok:
            ok = run(["nsys", "export", "--type=sqlite", "--force-overwrite=true", f"--output={profile}.sqlite", f"{profile}.nsys-rep"], name + "_export")
    elif setting == "stage_share":
        ok = run([sys.executable, STAGE_SHARE, "--batch_size", b, "--step", step], name)
    else:
        ok = run([sys.executable, HARNESS, "--setting", setting] + harness_args, name)
    if ok:
        results.append((name, "done"))
    else:
        results.append((name, "FAILED"))

#SUMMARY: every run and how it went, then the failed ones with where to find their output
summary = [f"FINISHED {time.strftime('%m-%d %H:%M:%S')}, {(time.time() - start_all) / 60:.0f} min", ""]
for name, status in results:
    summary.append(f"  {status:<8} {name}")
failed = []
for name, status in results:
    if status == "FAILED":
        failed.append(name)
summary.append("")
summary.append(f"{len(results) - len(failed)} of {len(results)} worked")
if failed:
    summary.append("FAILED, output of each one is in benchmarking/fill_logs/:")
    for name in failed:
        summary.append(f"  {name}.log")
else:
    summary.append("NOTHING FAILED")
print("\n" + "\n".join(summary))
(LOG_DIR / "summary.txt").write_text("\n".join(summary) + "\n") #same summary saved, in case the terminal is gone
