import re, subprocess, threading, time
import numpy as np, torch

INTERVALS = [10, 20, 50, 100]   # ms to test
DURATION = 10                   # seconds per interval
HALF_PERIOD = 0.05              # load on 50 ms, off 50 ms
LOAD_RE = re.compile(r"GR3D_FREQ (\d+)%")
PWR_RE = re.compile(r"(VDD_\w*GPU\w*|POM_5V_GPU) (\d+)(?:mW)?/")

stop = threading.Event()
def square_wave_load():
    a = torch.randn(2048, 2048, device="cuda")
    while not stop.is_set():
        t = time.perf_counter()
        while time.perf_counter() - t < HALF_PERIOD:   # busy
            a @ a; torch.cuda.synchronize()
        time.sleep(HALF_PERIOD)                         # idle

def measure(interval):
    p = subprocess.Popen(["stdbuf", "-oL", "tegrastats", "--interval", str(interval)],
                         stdout=subprocess.PIPE, text=True)
    stamps, load, pwr = [], [], []
    t_end = time.perf_counter() + DURATION
    for line in p.stdout:
        stamps.append(time.perf_counter())
        if m := LOAD_RE.search(line): load.append(int(m[1]))
        if m := PWR_RE.search(line): pwr.append(int(m[2]))
        if stamps[-1] > t_end: break
    p.terminate(); p.wait()
    gaps = np.diff(stamps) * 1000
    pwr_repeat = np.mean(np.diff(pwr) == 0) if len(pwr) > 1 else float("nan")
    print(f"{interval:>4} ms | actual {np.median(gaps):6.1f} ms (p90 {np.percentile(gaps, 90):6.1f}) | "
          f"lines {len(stamps):4d} | load min/med/max {min(load)}/{int(np.median(load))}/{max(load)} | "
          f"GPU power repeats {pwr_repeat:.0%}")

threading.Thread(target=square_wave_load, daemon=True).start()
time.sleep(2)  # let the load and clocks settle
for iv in INTERVALS:
    measure(iv)
stop.set()
