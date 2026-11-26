import sys
import csv
import argparse #cli args
import subprocess
from pathlib import Path #find paths relative to this script

from helper import run, latest_e2e_ms, avg, cost
from config import HARNESS_DIR, BENCH_DIR, STAGES, TIMED_LOOP, E2E_STEPS, TRACY_FLAGS

#TT version of the CUDA stage share analysis: finds the average percentage of each batch spent in every stage and substage
#of an ASYNC (no forced syncs) e2e run, then multiplies it by the unprofiled e2e batch time to estimate each substage's time
#USAGE: python main.py --batch_size 1 --step 16 [--optimization 1]
#       python main.py --batch_size 1 --reuse   (skip the runs, re-read the last saved profiles for this batch size)

#does:  1. runs the harness --setting e2e with no profiler             -> unprofiled ms per batch (from stage_timings.csv)
#       2. HOST run: same command under tracy --no-device              -> host ms of every substep (signposts only)
#       3. DEVICE run: same command under tracy with the card profiler -> device kernel ms of the card ops each substep dispatched
#       4. card timeline rebuilt on the HOST run's clock: every op the host run dispatched gets the device kernel duration of
#          the same op in the device run (matched one to one, in order, op names checked)
#       5. per batch: each substep costs max(host ms - overlap, card ms) (same rule as CUDA), its share = that / the batch's total

#NOTE: why TT needs two profiled runs (CUDA's nsys needs one)
#      with the card profiler on, tt-mlir's runtime reads the profiler data back after EVERY graph it runs
#      (ProgramExecutor::readProfilerDataIfNeeded(force = true) in tt-mlir runtime/lib/ttnn/program_executor.cpp),
#      which waits for the card. That run is ~30x slower and fully synchronous, so its host times are mostly profiler waiting.
#      its device kernel durations are still exact (the card times its own kernels), so:
#        host ms + op dispatch times come from the --no-device run (no card readbacks, stays close to the unprofiled run)
#        each op's device kernel duration comes from the card profiler run
#      the same graphs run the same ops in the same order every batch, so the two runs' ops line up one to one
#NOTE: card timeline rebuild (on the host run's clock): an op starts when the host dispatched it or when the card finished
#      the op before it, whichever is later, and runs for its kernel duration. Kernel durations leave out the small gaps
#      between ops on the card, so the card finishes slightly early -> overlap and card ms come out slightly low
#      card ms = first op start to last op end of the ops a substep dispatched (like nsys' GPU projection of an NVTX range)
#      overlap = part of a substep's host time while card work from earlier substeps was still running (same as CUDA)
#NOTE: tracy can't build its ops report without card data, so the host run "fails" at the very end; its signposts are read
#      from the host log tracy writes before that (<folder>/.logs/tracy_ops_data.csv), the failure is expected
#NOTE: no syncs are added, both profiled runs are the normal e2e run under tracy (the e2e pass already flushes each card
#      substep with torch_xla.sync(wait = False), see xla_yolo/timing.py)

#CLI ARGS
parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16, 32, 64] , default = 1, help = "determines throughput of images sent through the model per batch (defaulted to 1)")
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1)") #used to create accurate subset of db
parser.add_argument("--optimization", type = int, choices = [0, 1, 2], default = 0, help = "determines level of TT-XLA compiler optimization (defaulted to 0, same as the harness)")
parser.add_argument("--reuse", action = "store_true", help = "indicates that previous runs profiled data may be reused for calculation (defaulted to false, useful for demo)") #indicates if steps 1-3 are to be skipped
args = parser.parse_args()

batch_size = str(args.batch_size) #NOTE: string, since it goes on the command line and is compared to the csv's text
harness_args = ["main.py", "--setting", "e2e", "--batch_size", batch_size, "--step", str(args.step),
                "--optimization", str(args.optimization)] #args to be passed to harness run
host_dir = BENCH_DIR / f"tracy_e2e_b{batch_size}_host" #tracy --no-device writes .logs/tracy_ops_data.csv in here
device_dir = BENCH_DIR / f"tracy_e2e_b{batch_size}" #tracy writes reports/<date>/ops_perf_results_<date>.csv in here
host_log = host_dir / ".logs" / "tracy_ops_data.csv"

#helper function that returns the newest ops report tracy wrote into device_dir
def latest_ops_csv():
    found = sorted(device_dir.glob("reports/**/ops_perf_results*.csv"), key = lambda p: p.stat().st_mtime)
    if not found:
        sys.exit(f"ERROR: NO ops_perf_results*.csv FOUND UNDER {device_dir}/reports (did tracy run with -r?)")
    return found[-1]

#helper function that keeps only the events inside the LAST timed loop (warmup batches aren't counted)
#events: list of ("sp", name, ts) / ("op", ts, device ns) in time order
def timed_loop(events, source):
    starts = [i for i, e in enumerate(events) if e[0] == "sp" and e[1] == f"{TIMED_LOOP}_start"]
    if not starts:
        sys.exit(f"ERROR: NO {TIMED_LOOP}_start SIGNPOST FOUND IN {source}")
    end = next((i for i in range(starts[-1], len(events)) if events[i][0] == "sp" and events[i][1] == f"{TIMED_LOOP}_end"), len(events))
    return events[starts[-1] + 1 : end]

#helper function that returns an op's name from a tracy host message ('TT_DNN_DEVICE_OP: "Conv2dDeviceOperation", 123, 0, ...')
def op_name(msg):
    return msg.split(":", 1)[-1].split(",", 1)[0].strip().strip('"')

#helper function that walks the host run's signposts and card ops (each op already has its device duration) and returns
#one dict per batch: (stage, substep) -> [host ms, overlap ms, card ms]
#NOTE: card ops dispatched after a substep's _end but before the next substep starts are billed to that substep (its queued work)
def per_batch(events):
    batches = []
    stage, sub = None, None
    card_free = 0 #host-clock ns when the card finishes everything dispatched so far
    for e in events:
        if e[0] == "op":
            _, ts, dev = e
            op_start = max(ts, card_free) #starts once dispatched and once the card is free
            card_free = op_start + dev
            if sub is not None:
                rec = batches[-1][sub]
                if rec[5] is None: #first card op of this substep
                    rec[5] = op_start
                rec[6] = card_free #card end of this substep's ops so far
            continue
        _, name, ts = e
        if name.endswith("_start") and name[: -len("_start")] in STAGES: #stage opens
            stage, sub = name[: -len("_start")], None
            if stage == STAGES[0]:
                batches.append({})
        elif name.endswith("_end") and name[: -len("_end")] in STAGES: #stage closes
            stage, sub = None, None
        elif name.endswith("_start") and stage and batches: #substep opens
            sub = (stage, name[: -len("_start")])
            #[host start, host end, card done before it, -, -, first card op start, last card op end]
            batches[-1][sub] = [ts, None, card_free, None, None, None, None]
        elif name.endswith("_end") and sub and name[: -len("_end")] == sub[1]: #substep closes
            batches[-1][sub][1] = ts
    out = []
    for b in batches:
        rows = {}
        for key, (h_start, h_end, done_before, _, _, c_start, c_end) in b.items():
            if h_end is None: #substep never closed (cut off)
                continue
            overlap = max(0.0, min(h_end, done_before) - h_start)
            card = (c_end - c_start) if c_start is not None else 0.0
            rows[key] = ((h_end - h_start) / 1e6, overlap / 1e6, card / 1e6)
        if rows:
            out.append(rows)
    return out

if not args.reuse:
    #1. unprofiled run
    run("STEP 1: unprofiled e2e run", [sys.executable] + harness_args, HARNESS_DIR) #run harness in unprofiled (e2e) mode and capture e2e latency

    #2. host run: no card profiler. tracy's report step fails at the end without card data (expected, see NOTE), so no run() here
    print("\n    STEP 2: profiled e2e run under tracy, host only (--no-device)    ")
    cmd = [sys.executable, "-m", "tracy", "--no-device"] + TRACY_FLAGS + ["-o", str(host_dir)] + harness_args + ["--profiled"]
    print("  $ " + " ".join(cmd))
    if host_log.exists():
        host_log.unlink() #so an old log is never mistaken for this run's
    subprocess.run(cmd, cwd = HARNESS_DIR)
    if not host_log.exists():
        sys.exit(f"ERROR: STEP 2 WROTE NO HOST LOG ({host_log}), see the output above")
    print("  NOTE: the 'device log ... missing' error right above is expected for this host only run")

    #3. device run: card profiler on, only its device kernel durations are used
    run("STEP 3: profiled e2e run under tracy, with the card profiler", [sys.executable, "-m", "tracy"] + TRACY_FLAGS
                                                 + ["-o", str(device_dir)] + harness_args + ["--profiled"], HARNESS_DIR)
e2e_ms, e2e_step = latest_e2e_ms(BENCH_DIR / "stage_timings.csv", batch_size, E2E_STEPS)
ops_csv = latest_ops_csv()

#host run: signposts + card op dispatches, from tracy's host log (";" separated, messages wrapped in backticks)
host_events = [] #("sp", name, ts) / ("op", name, ts)
for row in csv.DictReader(open(host_log, newline = ""), delimiter = ";", quotechar = "`"):
    msg = row.get("MessageName") or ""
    if msg.startswith("TT_SIGNPOST"):
        host_events.append(("sp", msg.split(": ", 1)[-1].split("\n")[0].strip(), float(row["total_ns"])))
    elif msg.startswith("TT_DNN_DEVICE_OP"):
        host_events.append(("op", op_name(msg), float(row["total_ns"])))
host_events.sort(key = lambda e: e[2])
host_events = timed_loop(host_events, host_log)

#device run: card ops (name, device kernel duration) in dispatch order, from the ops report
dev_events = [] #("sp", name, ts) / ("op", name, ts, device ns)
for row in csv.DictReader(open(ops_csv)):
    try:
        if row.get("OP TYPE") == "signpost":
            dev_events.append(("sp", row["OP CODE"], float(row["HOST START TS"])))
        elif "device" in (row.get("OP TYPE") or "").lower():
            dev_events.append(("op", row["OP CODE"], float(row["HOST START TS"]), float(row.get("DEVICE KERNEL DURATION [ns]") or 0)))
    except (KeyError, ValueError): #if row is cut off or not an op
        pass #skip current line
dev_events.sort(key = lambda e: e[2]) #order by host time
dev_ops = [e for e in timed_loop(dev_events, ops_csv) if e[0] == "op"]

#match the two runs' ops one to one, in order, and give every host run op its device duration
host_ops = [e for e in host_events if e[0] == "op"]
if len(host_ops) != len(dev_ops):
    sys.exit(f"ERROR: HOST RUN HAS {len(host_ops)} CARD OPS IN THE TIMED LOOP, DEVICE RUN HAS {len(dev_ops)} (rerun both with the same args)")
mismatched = sum(1 for h, d in zip(host_ops, dev_ops) if h[1] != d[1])
if mismatched:
    sys.exit(f"ERROR: {mismatched} OF {len(host_ops)} OPS HAVE DIFFERENT NAMES IN THE TWO RUNS (rerun both with the same args)")
durations = iter(d[3] for d in dev_ops)
events = [("op", e[2], next(durations)) if e[0] == "op" else e for e in host_events]
missing_device = sum(1 for d in dev_ops if d[3] == 0) #card ops the card profiler didn't time, should be 0

batches = per_batch(events)
if not batches:
    sys.exit(f"ERROR: NO BATCHES FOUND (check STAGES match the signpost names)")

order = [] #(stage, substep) in the order they ran
shares, host, overlaps, card_t = {}, {}, {}, {}
for batch in batches:
    total = sum(cost(h, w, g) for h, w, g in batch.values())
    if total <= 0:
        continue
    for key, (h, w, g) in batch.items():
        if key not in shares:
            order.append(key)
            shares[key], host[key], overlaps[key], card_t[key] = [], [], [], []
        shares[key].append(cost(h, w, g) / total * 100)
        host[key].append(h)
        overlaps[key].append(w)
        card_t[key].append(g)

#display results in command line
#prints the average % of every substep (grouped by stage), times the unprofiled e2e time
bs = int(batch_size)
stage_ms = {} #stage -> est. ms per image
print(f"\n    RESULTS: batch {batch_size}, {len(batches)} batches, {len(host_ops)} card ops matched, "
      f"unprofiled e2e {e2e_ms:.3f} ms per batch ({e2e_step})    ")
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

print(f"\n  NOTE: per batch. host ms + overlap from the --no-device run, card ms = card timeline rebuilt with the device run's kernel times")
print(f"        cost = max(host ms - overlap, card ms) (same as CUDA), est. ms/img = avg % x unprofiled e2e per batch / batch size")
print(f"  CHECK: host run substeps {avg([sum(v[0] for v in b.values()) for b in batches]):.3f} ms per batch vs unprofiled e2e {e2e_ms:.3f}"
      f"   card ops with no device duration: {missing_device}")
print(f"\n  SAVED: {host_log}\n         {ops_csv}")
