import sys
import re
from pathlib import Path

#prints the average and max power, temperature and GPU load from tegrastats.log for each hardware run
#USAGE: python3 hw_simple.py [benchmarking folder]
#needs: tegrastats.log (lines start with the time.time() stamp), hw_metrics.csv, and hw_stages.csv files
#shows: the latest full pipeline run for each batch size, then the latest idle / pre / inference / post stage runs


folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

#read the log: keep (time, line) for every line that starts with a time stamp
readings = []
for line in open(folder / "tegrastats.log"):
    first = line.split(" ", 1)[0]
    try:
        readings.append((float(first), line))
    except ValueError: #if line doesn't have a timestamp
        pass  #skip current line

#jelper function that pulls the info out of a tegrastats log line
def parse(line):
    power = {} #rail name -> watts, from text like "VDD_GPU_SOC 5123mW/5000mW" (first number = right now)
    for name, mw in re.findall(r"(\w+) (\d+)mW/\d+mW", line):
        power[name] = int(mw) / 1000
    temp = {} #sensor name -> degrees C, from text like "gpu@41.2C"
    for name, c in re.findall(r"(\w+)@(-?[\d.]+)C", line):
        temp[name] = float(c)
    load = re.search(r"GR3D_FREQ (\d+)%", line) #GPU load
    return power, temp, int(load.group(1)) if load else None #returns power, temp, and GPU load

#print average and max of every reading inside one window [start, end]
def report(title, start, end):
    inside = [line for t, line in readings if start <= t <= end] #metrics found within the window of last run
    print(f"\n    {title}: {end - start:.1f} s, {len(inside)} readings    ")
    if not inside: #if no metrics found
        print("NO AVAILABLE LOG READINGS IN THIS TIME WINDOW")
        return
    powers, temps, loads = {}, {}, [] 
    #NOTE: powers and temps are constructed as dictionaries with keys of rail naems (e.g. "VDD_GPU_SOC")
    for line in inside: #fill dictionaries and loads list
        power, temp, load = parse(line)
        for k, v in power.items():
            powers.setdefault(k, []).append(v)
        for k, v in temp.items():
            temps.setdefault(k, []).append(v)
        if load is not None:
            loads.append(load)
    print(f"  {'POWER (W)':<16} {'avg':>7} {'max':>7}")
    for k in sorted(powers): #NOTE: loop over k rail names 
        v = powers[k]
        print(f"  {k:<16} {sum(v) / len(v):>7.2f} {max(v):>7.2f}")
    print(f"  {'TEMP (C)':<16} {'avg':>7} {'max':>7}")
    for k in sorted(temps):
        v = temps[k]
        print(f"  {k:<16} {sum(v) / len(v):>7.1f} {max(v):>7.1f}")
    if loads:
        print(f"  {'GPU LOAD (%)':<16} {sum(loads) / len(loads):>7.0f} {max(loads):>7}")

#full pipeline runs
#NOTE: hw_metrics.csv rows are monitor,phase,batch_size,start,end
latest = {} #batch size -> (start, end), later rows replace earlier ones
for line in open(folder / "hw_metrics.csv"):
    row = line.strip().split(",")
    if len(row) == 5 and row[0] == "tegrastats" and row[2].isdigit():
        latest[int(row[2])] = (float(row[3]), float(row[4]))
for b in sorted(latest):
    report(f"full pipeline, batch {b}", latest[b][0], latest[b][1])

#stage average performance from 5 seconds of runtime
#NOTE: hw_stages.csv rows are monitor,stage,(batch_size,)start,end,iterations
stages_path = folder / "hw_stages.csv"
if stages_path.exists():
    stages = {}
    for line in open(stages_path):
        row = line.strip().split(",")
        if row[0] == "tegrastats" and len(row) in (5, 6) and row[-1].isdigit():
            stages[row[1]] = (float(row[-3]), float(row[-2]), int(row[-1]))
    for name in ("idle", "pre_processing", "inference", "post_processing"):
        if name in stages:
            start, end, n = stages[name]
            report(f"stage {name} (batch 1, {(end - start) * 1000 / n:.2f} ms per run)", start, end)
