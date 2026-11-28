import sys
import subprocess

STAGES = ("pre_processing", "inference", "post_processing") #stages checked against idle (main.py imports this too)
SHOWN_ONLY = ("AICLK", "VCORE") #shown in main.py's table but not checked against idle (see NOTE at the top of main.py)

#HELPER FUNCTIONS
#NOTE: functions only see globals from the file they're defined in, so SRC_DIR / args are passed in as arguments

#helper function that executes one fresh hardware run and returns stage -> {metric: avg}
#hw_metrics prints a block per stage like:
#    stage idle (batch 1, step 4, opt 0, 10.06 ms per batch): 11-26 21:40:02, 5.0 s, 48 readings
#  POWER (W)            avg     max
#  board_power        38.20   39.00
#  power              21.40   22.00
#  TEMP (C)             avg     max
#  asic                45.3    45.5
#  AICLK (MHz)         1350    1350
#  VCORE (V)          0.825   0.826
#  CURRENT (A)        26.10   27.00
def fresh_run(SRC_DIR, args):
    result = subprocess.run([sys.executable, "analysis/hw_metrics/main.py", "--fresh", "--step", str(args.step), "--batch_size", str(args.batch_size),
                             "--optimization", str(args.optimization)], cwd = SRC_DIR, capture_output = True, text = True) #run from YOLO/src, text = True gives strings instead of bytes
    metrics = {} #stage -> {"POWER board_power": avg, "TEMP asic": avg, "AICLK": avg, "VCORE": avg, "CURRENT": avg}
    stage, section = None, None
    for line in result.stdout.splitlines():
        words = line.split()
        if not words:
            continue
        if line.startswith("    ") and words[0] in ("stage", "full"): #new block, only stage blocks are kept
            stage = words[1] if words[0] == "stage" else None
            if stage:
                metrics[stage] = {}
        elif stage is None:
            continue
        elif words[0] in ("POWER", "TEMP"): #table header, remember which table the next rows belong to
            section = words[0]
        elif words[0] in ("AICLK", "VCORE", "CURRENT"): #single line reading: name (unit) avg max
            metrics[stage][words[0]] = float(words[2]) #avg column
        elif section and len(words) == 3: #power / temp row: name avg max
            metrics[stage][f"{section} {words[0]}"] = float(words[1])
    #ERROR HANDLING OF SUBPROCESS
    missing = [s for s in ("idle",) + STAGES if not metrics.get(s)]
    if result.returncode != 0 or missing:
        print(result.stdout[-3000:]); print(result.stderr[-3000:]) #end of the run's output, where the error is
        sys.exit(f"ERROR: FRESH HARDWARE RUN FAILED OR HAD NO READINGS FOR: {', '.join(missing) or 'see above'}")
    return metrics

#helper function that returns every "stage metric" below idle in one run (temperatures within --temp_tolerance)
def below_idle(metrics, args):
    out = []
    for key, idle_v in metrics["idle"].items():
        if key in SHOWN_ONLY: #clock and voltage are set by the card's firmware, not by the work (see NOTE at the top of main.py)
            continue
        allowed = args.temp_tolerance if key.startswith("TEMP") else 0.0 #only temperatures get a tolerance (see NOTE at the top of main.py)
        for s in STAGES:
            v = metrics[s].get(key)
            if v is not None and v < idle_v - allowed:
                out.append(f"{s} {key}")
    return out
