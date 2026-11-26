import sys
import csv #open and read csvs
import argparse #cli args

import subprocess #runs the harness and nsys for you
from pathlib import Path #find paths relative to this script

#finds the average percentage of each batch spent in every stage and substage of an ASYNC (no forced syncs) e2e run,
#from nsys + the harness' NVTX ranges, then multiplies it by the unprofiled e2e batch time to estimate each substage's time
#USAGE: python3 stage_share.py <batch size> [any other main.py flags, e.g. --step 16 --optimization inductor]
#       python3 stage_share.py <batch size> --reuse   (skip the runs, re-read the last saved profile for this batch size)

#does:  1. runs main.py --setting e2e with no profiler            -> unprofiled ms per batch (from stage_timings.csv)
#       2. runs main.py --setting e2e --profiled under nsys       -> NVTX ranges + the GPU work launched inside each one
#       3. exports two nsys reports to csv:
#            nvtx_pushpop_trace   host (CPU) start/end of every range
#            nvtx_gpu_proj_trace  the same ranges projected onto the GPU: first to last GPU op launched inside the range
#       4. per batch: each substep costs max(host ms, GPU ms), its share = that / the batch's total
#          host ms leaves out the OVERLAP: host time while GPU work launched by earlier substeps was still running. That
#          time is already counted in the earlier substep's GPU ms, whether the CPU was waiting on it (NMS blocking until
#          inference finishes) or working behind it (launching neck kernels while backbone runs)
#NOTE: why both timelines: with no syncs the CPU only LAUNCHES GPU work, so a CPU range like "inference" closes as soon as
#      the kernels are queued. The GPU projection is where that work actually ran
#NOTE: why max: when the CPU is the bottleneck the GPU sits idle while the CPU prepares/launches the next substep, so the
#      substep takes its host time (this is also what pass 2's CUDA events measure, since they're recorded in the GPU
#      stream before and after the CPU work). When the GPU is the bottleneck it takes its GPU time
#NOTE: no syncs are added. nsys puts CPU ranges and GPU work on one clock, which is how the overlap is found
#NOTE: only ranges directly inside a stage count; ranges libraries push inside a substep (e.g. cub:: inside NMS) are skipped
#NOTE: TT version (tracy + signposts) is stage_share_tt.py, same output format

#PATHS
HARNESS_DIR = Path(__file__).parent.parent / "substage_compiled" #dir with main.py
BENCH_DIR = Path(__file__).parent.parent.parent / "benchmarking"  #dir with stage_timings.csv, profile reports saved here too

#NVTX MARKERS
STAGES = ("preprocess", "inference", "postprocess") #yolo stage markers
TIMED_LOOP = "timed_batch_loop" #only ranges inside the e2e batch are used
E2E_STEPS = ("BATCH_MEDIAN", "TOTAL_PER_BATCH") #e2e rows in stage_timings.csv to use, first one found wins

#CLI ARGS
parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
#NOTE: should be unncessary, only e2e runs
#parser.add_argument("--setting", choices = ["latency", "hardware", "holistic", "profile", "e2e"], default = "latency", help = "determines what setting harness runs under (defaulted to e2e)") 
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64] , default = 1, help = "determines throughput of images sent through the model per batch (defaulted to 1)")
#NOTE: should be unnecessary, only bf16 runs for now
#parser.add_argument("--precision", choices = ["bf16", "fp32"], default = "bf16", help = "determines model data percision (defaulted to bf16)") #model data precision, optional, bf16 default to best match TT 
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1)") #used to create accurate subset of db
#NOTE: should be unnecessary, only inductor runs for now
#parser.add_argument("--optimization", choices = ["inductor", "aot_eager", "eager"], default = "inductor", help = "determines torch.compile backend, aot_eager = no fusion / no Triton optimization (defaulted to inductor)") #set level of compile time optimization via compilation backend tuning
parser.add_argument("--reuse", action = "store_true", help = "indicates that previous runs profiled data may be reused for calculation (defaulted to false, useful for demo)") #indicates if step 1-3 are to be skipped, and only recalculates stale data
args = parser.parse_args()

harness_args = ["main.py", "--setting", "e2e", "--batch_size", args.batch_size, "--step", args.step] #args to be passed to harness run
profile = BENCH_DIR / f"nsys_e2e_b{batch_size}" #nsys adds .nsys-rep, nsys stats adds _<report>.csv
host_csv = Path(f"{profile}_nvtx_pushpop_trace.csv")
gpu_csv = Path(f"{profile}_nvtx_gpu_proj_trace.csv")

#HELPER FUNCTIONS
#helper function that runs one command from the harness folder and stops everything if it fails
def run(title, cmd):
    print(f"\n    {title}    ")
    print("  $ " + " ".join(cmd)) #show the exact command so it can be rerun by hand
    if subprocess.run(cmd, cwd = HARNESS_DIR).returncode != 0:
        sys.exit(f"ERROR: {title} FAILED (see the output above)")

#helper function that returns the latest unprofiled e2e ms per batch saved in stage_timings.csv for this batch size
def latest_e2e_ms():
    found = {}
    for row in csv.DictReader(open(BENCH_DIR / "stage_timings.csv")):
        if row.get("pass") == "e2e" and row.get("batch_size") == batch_size and row.get("step") in E2E_STEPS:
            found[row["step"]] = float(row["ms"]) #later rows replace earlier ones
    for step in E2E_STEPS:
        if step in found:
            return found[step], step
    sys.exit(f"ERROR: NO e2e {' / '.join(E2E_STEPS)} ROW FOR BATCH {batch_size} IN stage_timings.csv")

#helper function that reads an nsys stats csv into a list of dicts
#NOTE: nsys prints a few lines before the header in some versions, so the header is found by looking for "Name"
def read_nsys_csv(path):
    lines = open(path).read().splitlines()
    first = next((i for i, line in enumerate(lines) if "Name" in next(csv.reader([line]), [])), None)
    if first is None:
        sys.exit(f"ERROR: NO HEADER ROW IN {path}")
    return list(csv.DictReader(lines[first:]))

#helper function that strips the NVTX domain nsys puts in front of every name (":A_LetterBox" -> "A_LetterBox")
def clean(name):
    return name.split(":", 1)[-1] if name else name

#helper function that returns the value of the first column whose header starts with one of the given names
def col(row, *names):
    for key, value in row.items():
        if key and any(key.startswith(n) for n in names):
            return value
    return None

if not reuse:
    #1. unprofiled run
    run("STEP 1: unprofiled e2e run", [sys.executable] + harness_args)

    #2. profiled run (same command under nsys, plus --profiled)
    run("STEP 2: profiled e2e run under nsys", ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                                                 "--force-overwrite=true", "-o", str(profile), sys.executable]
                                                 + harness_args + ["--profiled"])

    #3. export host ranges and their GPU projection to csv
    host_csv.unlink(missing_ok = True) #remove old exports so nsys stats doesn't stop to ask about them
    gpu_csv.unlink(missing_ok = True)
    Path(f"{profile}.sqlite").unlink(missing_ok = True) #nsys stats converts the profile to sqlite first and won't overwrite an old one
    run("STEP 3: exporting NVTX ranges", ["nsys", "stats", "--force-export=true", "--report", "nvtx_pushpop_trace",
                                          "--report", "nvtx_gpu_proj_trace", "--format", "csv", "--output", str(profile),
                                          f"{profile}.nsys-rep"])
elif not (host_csv.exists() and gpu_csv.exists()):
    sys.exit(f"ERROR: --reuse BUT NO SAVED EXPORTS FOR BATCH {batch_size} ({host_csv.name}, {gpu_csv.name})")
e2e_ms, e2e_step = latest_e2e_ms()

#read host ranges: (name, start, end) in ns
ranges = []
for row in read_nsys_csv(host_csv):
    try:
        ranges.append((clean(col(row, "Name")), float(col(row, "Start")), float(col(row, "End"))))
    except (TypeError, ValueError): #if row is cut off or not a range
        pass #skip current line
ranges.sort(key = lambda r: r[1]) #order by start time

#read GPU projections: (name, host start) -> (GPU ms, GPU end ns). Ranges that launched no GPU work aren't in this report
gpu = {}
for row in read_nsys_csv(gpu_csv):
    try:
        key = (clean(col(row, "Name")), float(col(row, "Orig Start")))
        start, dur = float(col(row, "Projected Start")), float(col(row, "Projected Duration"))
        gpu[key] = (dur / 1e6, start + dur)
    except (TypeError, ValueError):
        pass

#only keep ranges inside the timed loop so warmup batches aren't counted
loop = [r for r in ranges if r[0] == TIMED_LOOP]
if not loop:
    sys.exit(f"ERROR: NO {TIMED_LOOP} RANGE FOUND (was the run --setting e2e?)")
loop_start, loop_end = loop[-1][1], loop[-1][2] #latest timed loop in the file
ranges = [r for r in ranges if loop_start <= r[1] and r[2] <= loop_end and r[0] != TIMED_LOOP]

#split into batches (a new batch starts at every first-stage range) and give every substep its stage
batches = [] #list of batches, each a list of (stage, substep, host ms, overlap ms, GPU ms)
stage, stage_end, sub_end = None, 0, 0
gpu_done = 0 #ns when the GPU finished everything launched so far (on the same clock as the CPU ranges)
for name, start, end in ranges:
    if name in STAGES:
        if name == STAGES[0]:
            batches.append([])
        stage, stage_end = name, end
        continue
    if not (batches and stage and start < stage_end): #not inside a stage
        continue
    if start < sub_end: #nested inside the previous substep (a library's own range), already counted in that substep
        continue
    sub_end = end
    g_ms, g_end = gpu.get((name, start), (0.0, 0))
    overlap = max(0.0, min(end, gpu_done) - start) #part of this range while earlier GPU work was still running
    batches[-1].append((stage, name, (end - start) / 1e6, overlap / 1e6, g_ms))
    gpu_done = max(gpu_done, g_end)
if not batches:
    sys.exit(f"ERROR: NO BATCHES FOUND (check STAGES match the range names in {host_csv})")

#percentage of each batch spent in every substep
#NOTE: a substep costs max(host time minus overlap with earlier GPU work, GPU time), see the top of the file
def cost(h, w, g):
    return max(h - w, g)

order = [] #(stage, substep) in the order they ran
shares, host, overlaps, gpu_t = {}, {}, {}, {}
for batch in batches:
    total = sum(cost(h, w, g) for _, _, h, w, g in batch)
    if total <= 0:
        continue
    for st, name, h, w, g in batch:
        key = (st, name)
        if key not in shares:
            order.append(key)
            shares[key], host[key], overlaps[key], gpu_t[key] = [], [], [], []
        shares[key].append(cost(h, w, g) / total * 100)
        host[key].append(h)
        overlaps[key].append(w)
        gpu_t[key].append(g)

def avg(v):
    return sum(v) / len(v) if v else 0.0

#print the average % of every substep (grouped by stage), times the unprofiled e2e time
n = len(batches)
bs = int(batch_size)
print(f"\n    RESULTS: batch {batch_size}, {n} batches, unprofiled e2e {e2e_ms:.3f} ms per batch ({e2e_step})    ")
print(f"  {'STAGE':<16} {'SUBSTEP':<22} {'host ms':>8} {'overlap':>8} {'GPU ms':>8} {'avg %':>7} {'est. ms/img':>12}")
for st in STAGES:
    stage_pct = 0.0
    for key in [k for k in order if k[0] == st]:
        pct = avg(shares[key])
        stage_pct += pct
        print(f"  {st:<16} {key[1]:<22} {avg(host[key]):>8.3f} {avg(overlaps[key]):>8.3f} {avg(gpu_t[key]):>8.3f} {pct:>7.2f} "
              f"{pct / 100 * e2e_ms / bs:>12.3f}")
    print(f"  {st:<16} {'STAGE TOTAL':<22} {'':>8} {'':>8} {'':>8} {stage_pct:>7.2f} {stage_pct / 100 * e2e_ms / bs:>12.3f}")
print(f"  NOTE: host / overlap / GPU ms are from the profiled run, per batch. overlap = host time while earlier GPU work still ran")
print(f"        est. ms/img = avg % x unprofiled e2e per batch / batch size")
print(f"\n  SAVED: {profile}.nsys-rep  (open in Nsight Systems to see the timeline)")
print(f"  SAVED: {host_csv}")
print(f"  SAVED: {gpu_csv}")
