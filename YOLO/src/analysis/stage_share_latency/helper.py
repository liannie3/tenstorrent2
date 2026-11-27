import sys
import csv
import subprocess

#HELPER FUNCTIONS
#NOTE: functions only see globals from the file they're defined in, so paths / batch size are passed in as arguments
#helper function that runs one command from the given folder and stops everything if it fails
def run(title, cmd, cwd):
    print(f"\n    {title}    ")
    print("  $ " + " ".join(cmd)) #show the exact command so it can be rerun by hand
    if subprocess.run(cmd, cwd = cwd).returncode != 0:
        sys.exit(f"ERROR: {title} FAILED (see the output above)")

#helper function that returns the latest unprofiled e2e ms per batch saved in stage_timings.csv for this batch size
#NOTE: batch_size is a string, the csv stores everything as text
def latest_e2e_ms(csv_path, batch_size, steps):
    found = {}
    for row in csv.DictReader(open(csv_path)):
        substep = row.get("substep", row.get("step")) #"substep" column (was "step" in older stage_timings.csv files)
        if row.get("pass") == "e2e" and row.get("batch_size") == batch_size and substep in steps:
            found[substep] = float(row["ms"]) #later rows replace earlier ones
    for step in steps:
        if step in found:
            return found[step], step
    sys.exit(f"ERROR: NO e2e {' / '.join(steps)} ROW FOR BATCH {batch_size} IN {csv_path}")

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

#helper function that returns the average of a list (0 for an empty list)
def avg(v):
    return sum(v) / len(v) if v else 0.0

#helper function that returns how long a substep costs in a batch
#NOTE: max(host time minus overlap with earlier GPU work, GPU time), see the top of stage_share.py
def cost(h, w, g):
    return max(h - w, g)

#TT (tracy) HELPERS
#NOTE: events are tuples in host time order: ("sp", signpost name, ts) for signposts, ("op", ...) for card ops
#helper function that returns the newest ops report tracy wrote into the given folder
def latest_ops_csv(folder):
    found = sorted(folder.glob("reports/**/ops_perf_results*.csv"), key = lambda p: p.stat().st_mtime)
    if not found:
        sys.exit(f"ERROR: NO ops_perf_results*.csv FOUND UNDER {folder}/reports (did tracy run with -r?)")
    return found[-1]

#helper function that keeps only the events inside the LAST timed loop (warmup batches aren't counted)
def timed_loop(events, source, loop_name):
    starts = [i for i, e in enumerate(events) if e[0] == "sp" and e[1] == f"{loop_name}_start"]
    if not starts:
        sys.exit(f"ERROR: NO {loop_name}_start SIGNPOST FOUND IN {source}")
    end = next((i for i in range(starts[-1], len(events)) if events[i][0] == "sp" and events[i][1] == f"{loop_name}_end"), len(events))
    return events[starts[-1] + 1 : end]

#helper function that returns an op's name from a tracy host message ('TT_DNN_DEVICE_OP: "Conv2dDeviceOperation", 123, 0, ...')
def op_name(msg):
    return msg.split(":", 1)[-1].split(",", 1)[0].strip().strip('"')

#helper function that walks the host run's signposts and card ops (each op already has its device duration) and returns
#one dict per batch: (stage, substep) -> [host ms, overlap ms, card ms]
#NOTE: card ops dispatched after a substep's _end but before the next substep starts are billed to that substep (its queued work)
def per_batch(events, stages):
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
                if rec[3] is None: #first card op of this substep
                    rec[3] = op_start
                rec[4] = card_free #card end of this substep's ops so far
            continue
        _, name, ts = e
        if name.endswith("_start") and name[: -len("_start")] in stages: #stage opens
            stage, sub = name[: -len("_start")], None
            if stage == stages[0]:
                batches.append({})
        elif name.endswith("_end") and name[: -len("_end")] in stages: #stage closes
            stage, sub = None, None
        elif name.endswith("_start") and stage and batches: #substep opens
            sub = (stage, name[: -len("_start")])
            #[host start, host end, card done before it, first card op start, last card op end]
            batches[-1][sub] = [ts, None, card_free, None, None]
        elif name.endswith("_end") and sub and name[: -len("_end")] == sub[1]: #substep closes
            batches[-1][sub][1] = ts
    out = []
    for b in batches:
        rows = {}
        for key, (h_start, h_end, done_before, c_start, c_end) in b.items():
            if h_end is None: #substep never closed (cut off)
                continue
            overlap = max(0.0, min(h_end, done_before) - h_start)
            card = (c_end - c_start) if c_start is not None else 0.0
            rows[key] = ((h_end - h_start) / 1e6, overlap / 1e6, card / 1e6)
        if rows:
            out.append(rows)
    return out
