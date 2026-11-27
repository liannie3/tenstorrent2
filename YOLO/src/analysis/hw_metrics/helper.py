import re
import time

#HELPER FUNCTIONS

#returns  the desired info out of a tegrastats log line
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
#NOTE: requires readings to be passed from main.py
def report(title, start, end, readings):
    inside = [line for t, line in readings if start <= t <= end] #metrics found within the window of the run
    print(f"\n    {title}: {time.strftime('%m-%d %H:%M:%S', time.localtime(start))}, {end - start:.1f} s, {len(inside)} readings    ")
    if not inside: #if no metrics found
        print("NO AVAILABLE LOG READINGS IN THIS TIME WINDOW")
        return
    powers, temps, loads = {}, {}, []
    #NOTE: powers and temps are dictionaries keyed by rail / sensor names (e.g. "VDD_GPU_SOC")
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
