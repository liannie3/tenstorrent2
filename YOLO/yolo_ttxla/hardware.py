import torch
import time
from contextlib import contextmanager
import importlib.util #checks the tt-smi python packages are installed
import subprocess #execute tegrastats during hardware metric run
import sys

import config
import results
import run
from run import run_pipeline
from stages import preprocess_stage, inference_stage, postprocess_stage
from timing import _card_sync #requires read back barrier to repalce torch.cuda.synchronize()

#hardware mnitor for the tt card, equivalent to `tegratast --interval 100` on CUDA
#each tt smi snapshot takes 190ms, so instead we open the card once with tt-smi library and read sensors in a loop
#printing and writing metrics to a CSV row every INTERVAL ms (columns in TT_SMI_FIELDS)


TT_SAMPLER = """
import sys, time
from tt_umd import TelemetryTag
from tt_smi.utils import discover   #opens the card exactly the way `tt-smi -s` does (already tested alongside the benchmark)

interval = float(sys.argv[1]) / 1000
_, devices = discover()
reader = list(devices.values())[0].get_arc_telemetry_reader()
TAG_INPUT_POWER = 54   #whole-card power; tt-smi reads it by this number too

def temp_c(raw):       #temperature is stored as a signed 16.16 fixed-point number
    if raw & 0x80000000:
        raw -= 1 << 32
    return raw / 65536

while True:
    print(f"{time.time():.4f},"
          f"{reader.read_entry(TelemetryTag.TDP.value) & 0xFFFF},"
          f"{reader.read_entry(TAG_INPUT_POWER)},"
          f"{reader.read_entry(TelemetryTag.AICLK.value) & 0xFFFF},"
          f"{reader.read_entry(TelemetryTag.VCORE.value) / 1000},"
          f"{reader.read_entry(TelemetryTag.TDC.value) & 0xFFFF},"
          f"{temp_c(reader.read_entry(TelemetryTag.ASIC_TEMPERATURE.value)):.2f}", flush = True)
    time.sleep(interval)
"""

TT_SMI_FIELDS = "time,power_w,board_power_w,aiclk_mhz,voltage_v,current_a,asic_temp_c"


def identify_monitor():
    #check for tt-smi
    if importlib.util.find_spec("tt_smi") and importlib.util.find_spec("tt_umd"): 
        return "tt_smi", [sys.executable, "-c", TT_SAMPLER, "100"]
    #NOTE: telemetry is update every 100 ms following UPDATE_TELEM_SPEED = 0x64, so no point on polling tighter than 100ms
        
    return None

#starts the hardware monitor in the background, sets the hardware flags, and always cleans both up
@contextmanager
def hw_monitor():
    found_monitor = identify_monitor() #chooses which hw monitor is used for hw runs
    if found_monitor is None:
        raise RuntimeError("NO HARDWARE MONITOR FOUND")
    name, cmd = found_monitor

    config.SUBSTEP_TIMING = False
    config.HARDWARE_METRICS = True
    config.BENCH_DIR.mkdir(parents = True, exist_ok = True) #create benchmarking/ folder if missing
    log_path = config.BENCH_DIR / f"{name}.csv" #write to tt_smi.csv vs log files in CUDA

    if not log_path.exists(): #write the csv header once the first time the file is made
        with open(log_path, "w") as log:
            log.write(TT_SMI_FIELDS + "\n")
                
    with open(log_path, "a") as log:
        proc = subprocess.Popen(cmd, stdout = log)
        try:
            time.sleep(1) #let the monitor start
            yield name
        finally: #runs even if the pass crashes
            proc.terminate()
            proc.wait()
            config.HARDWARE_METRICS = False

########################################################################################################################################################################
#                         | | | |                                                                                                                                      #
#   Run Pipeline and Main V V V V                                                                                                                                      #
########################################################################################################################################################################

@torch.no_grad()
def hardware_run_pipeline(model, img_paths, batch_size : int = config.BATCH_SIZE, device = config.DEVICE):
    with hw_monitor() as monitor:
        run.run_pipeline(model, img_paths, batch_size, device)

    results.store_hw_window(monitor, "full_pipeline", run.hw_window["start"], run.hw_window["end"])

#test each substage in isolation over a fixed period and accumulates, averages, then stores associated hardware metrics
@torch.no_grad()
def hardware_substages(model, images, device = config.DEVICE):
    #prepare each stages inputs before metrics are gathered
    sample = images[:config.HW_SAMPLE_SIZE]
    ims, shapes, preds = [], [], []
    for i in range(len(sample)):
        im, shape = preprocess_stage(sample[i:i+1], device)
        ims.append(im)
        shapes.append(shape)
        preds.append(inference_stage(model, im, device))
    _card_sync()

    #lookup table to invoke each substage (and an idle for baseline) independently
    substages = {
        "idle": lambda i: time.sleep(0.01),
        "pre_processing": lambda i: preprocess_stage(sample[i:i+1], device),
        "inference": lambda i: inference_stage(model, ims[i], device),
        "post_processing": lambda i: postprocess_stage(preds[i], ims[i], shapes[i], config.CONF_THRES, config.IOU_THRESH, device),
    }

    #warm every substage up before the monitor start to avoid compile time overhead appearing inside the pre_processing and inference windows
    for substage_func in substages.values():
        for _ in range(config.COMPILE_WARMUP):
            substage_func(0)

    runs = []
    with hw_monitor() as monitor:
        for name, substage_func in substages.items(): #grab each substage from the lookup tablew
            _card_sync()
            t_start = time.time()
            n = 0 #reset to only track iterations within a run
            while (time.time() - t_start) < config.PHASE_LIMIT: #run the substage under a time limit
                substage_func(n % len(sample) ) #iterate through sample images
                n+=1
            _card_sync() #force cpu to wait for card work after runs finish
            
            t_end = time.time()
            runs.append({"monitor": monitor,"stage": name, "start" : t_start, "end" : t_end, "iterations" : n})
            print(f"{name:<16} {n} runs, {(t_end - t_start) / n * 1000:.2f} ms per run")
            time.sleep(1) #further discretize phases of substage testing

    results.store_hw_stages(runs)
