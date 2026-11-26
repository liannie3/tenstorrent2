import sys
import csv #open and read csvs
import argparse #cli args
from pathlib import Path #find paths relative to this script

from helper import run, latest_e2e_ms, avg, cost
from config import HARNESS_DIR, BENCH_DIR, STAGES, TIMED_LOOP, E2E_STEPS, TRACY_FLAGS

#TT version of the CUDA stage share analysis: same steps, same cost rule, same output format
#finds the average percentage of each batch spent in every stage and substage of an ASYNC (no forced syncs) e2e run,
#from tracy + the harness' signposts, then multiplies it by the unprofiled e2e batch time to estimate each substage's time
#USAGE: python3 main.py --batch_size 1 --step 16 [--optimization 1]
#       python3 main.py --batch_size 1 --reuse   (skip the runs, re-read the last saved ops report for this batch size)

#does:  1. runs the harness --setting e2e with no profiler          -> unprofiled ms per batch (from stage_timings.csv)
#       2. runs it again --profiled under tracy                    -> ops report csv: one row per signpost (host time) and
#                                                                      one row per card op (host dispatch time + device duration)
#       3. per batch: each substep costs max(host ms - overlap, card ms), its share = that / the batch's total (same as CUDA)

#NOTE: differences from the CUDA (nsys) version, and why
#      host ms:  time between a substep's "<name>_start" and "<name>_end" signposts (nsys: the NVTX range)
#      card ms:  span of the card ops dispatched between those signposts (nsys: GPU projection of the range)
#      tracy reports each op's device duration, but not on the host's clock (the card counts its own cycles). So the card
#      timeline is rebuilt on the host clock: an op starts when the host finished dispatching it, or when the card finished
#      the op before it, whichever is later, and runs for its device duration. Overlap is then found exactly like on CUDA
#      (the rebuild ignores the few microseconds between dispatch and start, so card work ends slightly early -> slightly
#      less overlap than reality)
#      card ops dispatched between two substeps (e.g. by run.py's torch_xla.sync(wait = False) at the end of a stage) are
#      billed to the substep before them, since they are that substep's queued work

#CLI ARGS
parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64] , default = 1, help = "determines throughput of images sent through the model per batch (defaulted to 1)")
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1)") #used to create accurate subset of db
parser.add_argument("--optimization", type = int, choices = [0, 1, 2], default = 0, help = "determines level of TT-XLA compiler optimization (defaulted to 0, same as the harness)")
parser.add_argument("--reuse", action = "store_true", help = "indicates that previous runs profiled data may be reused for calculation (defaulted to false, useful for demo)") #indicates if steps 1-2 are to be skipped
args = parser.parse_args()

batch_size = str(args.batch_size) #NOTE: string, since it goes on the command line and is compared to the csv's text
harness_args = ["main.py", "--setting", "e2e", "--batch_size", batch_size, "--step", str(args.step),
                "--optimization", str(args.optimization)] #args to be passed to harness run
profile_dir = BENCH_DIR / f"tracy_e2e_b{batch_size}" #tracy writes reports/<date>/ops_perf_results_<date>.csv in here

#helper function that returns the newest ops report tracy wrote into profile_dir
def latest_ops_csv():
    found = sorted(profile_dir.glob("reports/**/ops_perf_results*.csv"), key = lambda p: p.stat().st_mtime)
    if not found:
        sys.exit(f"ERROR: NO ops_perf_results*.csv FOUND UNDER {profile_dir}/reports (did tracy run with -r?)")
    return found[-1]

if not args.reuse:
    #1. unprofiled run
    run("STEP 1: unprofiled e2e run", [sys.executable] + harness_args, HARNESS_DIR) #run harness in unprofiled (e2e) mode and capture e2e latency

    #2. profiled run (same command under tracy, plus --profiled)
    run("STEP 2: profiled e2e run under tracy", [sys.executable, "-m", "tracy"] + TRACY_FLAGS + ["-o", str(profile_dir)]
                                                 + harness_args + ["--profiled"], HARNESS_DIR) #tracy builds the ops report at the end
e2e_ms, e2e_step = latest_e2e_ms(BENCH_DIR / "stage_timings.csv", batch_size, E2E_STEPS)
ops_csv = latest_ops_csv()

#read the ops report: signposts -> (name, host ns), card ops -> (host dispatch start ns, host dispatch end ns, device ns)
#NOTE: rows are already in host time order; finds columns by header name so the exact column order doesn't matter
events = [] #("sp", name, ts) or ("op", host start, host end, device ns or None)
for row in csv.DictReader(open(ops_csv)):
    try:
        if row.get("OP TYPE") == "signpost":
            events.append(("sp", row["OP CODE"], float(row["HOST START TS"])))
        elif row.get("HOST START TS"):
            dev = row.get("DEVICE KERNEL DURATION [ns]") or row.get("DEVICE FW DURATION [ns]")
            if not dev and "device" not in (row.get("OP TYPE") or "").lower():
                continue #host-only rows have no device duration
            h_start = float(row["HOST START TS"])
            events.append(("op", h_start, float(row.get("HOST END TS") or h_start), float(dev) if dev else None))
    except (KeyError, ValueError): #if row is cut off or not an op
        pass #skip current line
events.sort(key = lambda e: e[2] if e[0] == "sp" else e[1]) #order by host time

#only keep events inside the last timed loop so warmup batches aren't counted
loop_starts = [i for i, e in enumerate(events) if e[0] == "sp" and e[1] == f"{TIMED_LOOP}_start"]
if not loop_starts:
    sys.exit(f"ERROR: NO {TIMED_LOOP}_start SIGNPOST FOUND IN {ops_csv}")
first = loop_starts[-1]
last = next((i for i in range(first, len(events)) if events[i][0] == "sp" and events[i][1] == f"{TIMED_LOOP}_end"), len(events))
events = events[first + 1 : last]

#rebuild the card timeline on the host clock (see the NOTE at the top) and walk the signposts
#batches: list of batches, each a list of [stage, substep, host ms, overlap ms, card ms]
batches = []
stage = None #stage currently open
sub = None #substep currently open (or the last one in this stage, for ops dispatched between substeps)
sub_start = 0 #host ns the open substep started
card_free = 0 #host-clock ns when the card finishes everything dispatched so far
card_done_at_start = 0 #card_free when the open substep started (earlier substeps' card work)
outside = 0 #card ops dispatched outside every stage (should be ~0)
missing_device = 0 #card ops with no device duration (on-card profiler buffer full, or op not profiled), should be 0
for e in events:
    if e[0] == "op":
        _, h_start, h_end, dev = e
        if dev is None:
            missing_device += 1
            continue
        op_start = max(h_end, card_free) #starts once dispatched and once the card is free
        card_free = op_start + dev
        if sub is None:
            outside += 1
            continue
        if sub[5] is None: #first card op of this substep
            sub[5] = op_start
        sub[6] = card_free #card end of this substep's ops so far
        continue
    _, name, ts = e
    if name.endswith("_start") and name[: -len("_start")] in STAGES: #stage opens
        stage = name[: -len("_start")]
        if stage == STAGES[0]:
            batches.append([])
        sub = None
    elif name.endswith("_end") and name[: -len("_end")] in STAGES: #stage closes
        stage, sub = None, None
    elif name.endswith("_start") and stage and batches: #substep opens inside a stage
        #[stage, substep, host start, host end, card done before it, first card op start, last card op end]
        sub = [stage, name[: -len("_start")], ts, None, card_free, None, None]
        batches[-1].append(sub)
    elif name.endswith("_end") and sub and name[: -len("_end")] == sub[1]: #substep closes
        sub[3] = ts #ops dispatched after this but before the next substep still go to this substep (see NOTE)

#turn the raw substeps into (stage, substep, host ms, overlap ms, card ms), the same shape as the CUDA version
for b in range(len(batches)):
    rows = []
    for st, name, h_start, h_end, done_before, c_start, c_end in batches[b]:
        if h_end is None: #substep never closed (cut off)
            continue
        overlap = max(0.0, min(h_end, done_before) - h_start) #part of this substep while earlier card work was still running
        card = (c_end - c_start) if c_start is not None else 0.0
        rows.append((st, name, (h_end - h_start) / 1e6, overlap / 1e6, card / 1e6))
    batches[b] = rows
batches = [b for b in batches if b]
if not batches:
    sys.exit(f"ERROR: NO BATCHES FOUND (check STAGES match the signpost names in {ops_csv})")

order = [] #(stage, substep) in the order they ran
shares, host, overlaps, card_t = {}, {}, {}, {}
for batch in batches:
    total = sum(cost(h, w, g) for _, _, h, w, g in batch)
    if total <= 0:
        continue
    for st, name, h, w, g in batch:
        key = (st, name)
        if key not in shares:
            order.append(key)
            shares[key], host[key], overlaps[key], card_t[key] = [], [], [], []
        shares[key].append(cost(h, w, g) / total * 100)
        host[key].append(h)
        overlaps[key].append(w)
        card_t[key].append(g)

#display results in command line
#prints the average % of every substep (grouped by stage), times the unprofiled e2e time
n = len(batches)
bs = int(batch_size)
stage_ms = {} #stage -> est. ms per image
print(f"\n    RESULTS: batch {batch_size}, {n} batches, unprofiled e2e {e2e_ms:.3f} ms per batch ({e2e_step})    ")
print(f"  {'STAGE':<16} {'SUBSTEP':<22} {'host ms':>8} {'overlap':>8} {'card ms':>8} {'avg %':>7} {'est. ms/img':>12}")
for st in STAGES:
    stage_pct = 0.0
    for key in [k for k in order if k[0] == st]:
        pct = avg(shares[key])
        stage_pct += pct
        print(f"  {st:<16} {key[1]:<22} {avg(host[key]):>8.3f} {avg(overlaps[key]):>8.3f} {avg(card_t[key]):>8.3f} {pct:>7.2f} "
              f"{pct / 100 * e2e_ms / bs:>12.3f}")
    stage_ms[st] = stage_pct / 100 * e2e_ms / bs #store this stage's est. ms per image
    print(f"  {st:<16} {'STAGE TOTAL':<22} {'':>8} {'':>8} {'':>8} {stage_pct:>7.2f} {stage_ms[st]:>12.3f}")

#per image and per batch totals of every stage
print()
for st in STAGES:
    print(f"  {st:<16} EST. TIME PER IMAGE {stage_ms[st]:>10.3f} ms   PER BATCH {stage_ms[st] * bs:>10.3f} ms")

print(f"\n  NOTE: host / overlap / card ms are from the profiled run, per batch. overlap = host time while earlier card work still ran")
print(f"        est. ms/img = avg % x unprofiled e2e per batch / batch size")
#sanity checks: both should be 0, otherwise some card work isn't counted
print(f"  CHECK: card ops outside every stage: {outside}   card ops with no device duration: {missing_device}")
print(f"\n  SAVED: {ops_csv}  (every op and signpost used above)")
