import sys
import csv
import subprocess

#HELPER FUNCTIONS

#runs one command from the given folder and stops everything if it fails
def run(title, cmd, cwd):
    print(f"\n    {title}    ")
    print("  $ " + " ".join(cmd)) #show the exact command so it can be rerun by hand
    if subprocess.run(cmd, cwd = cwd).returncode != 0:
        sys.exit(f"ERROR: {title} FAILED (see the output above)")

#returns the latest unprofiled e2e ms per batch saved in stage_timings.csv for this batch size and step size
def latest_e2e_ms(csv_path, batch_size, steps, step):
    found = {}
    for row in csv.DictReader(open(csv_path)):
        if "substep" not in row: #older stage_timings.csv rows, before the "step" column was renamed to "substep"
            continue
        if row.get("pass") == "e2e" and row.get("batch_size") == batch_size and row.get("step") == step and row.get("substep") in steps:
            found[row["substep"]] = float(row["ms"]) #later rows replace earlier ones
    for substep in steps:
        if substep in found:
            return found[substep], substep
    sys.exit(f"ERROR: NO e2e {' / '.join(steps)} ROW FOR BATCH {batch_size} STEP {step} IN {csv_path}")

#reads an nsys stats csv into a list of dicts
#NOTE: nsys prints a few lines before the header in some versions, so the header is found by looking for "Name"
def read_nsys_csv(path):
    lines = open(path).read().splitlines()
    first = next((i for i, line in enumerate(lines) if "Name" in next(csv.reader([line]), [])), None)
    if first is None:
        sys.exit(f"ERROR: NO HEADER ROW IN {path}")
    return list(csv.DictReader(lines[first:]))

#strips the NVTX domain nsys puts in front of every name (":A_LetterBox" -> "A_LetterBox")
def clean(name):
    return name.split(":", 1)[-1] if name else name

#returns the value of the first column whose header starts with one of the given stage names
def col(row, *stages):
    for key, value in row.items():
        if key and any(key.startswith(n) for n in stages):
            return value
    return None

#returns the average of a list (0 for an empty list)
def avg(v):
    return sum(v) / len(v) if v else 0.0

#returns how long a substep costs in a batch
#chooses either the time on gpu, or host time that didn't overlap previous gpu work (i.e. whcihver side was bottleneck for the substep)
def cost(host_time, overlap, gpu_time):
    return max(host_time - overlap, gpu_time)
