import torch
import time
from contextlib import contextmanager                       
import shutil #detecting correct hardware monitor
import subprocess #execute tegrastats during hardware metric run
import threading #reads tegrastats in the background while benchmark runs
import sys
import config
import results
import run
from run import run_pipeline
from stages import preprocess_stage, inference_stage, postprocess_stage


def timestamp(proc, log): #add timestamps with finer granularity than tegrastats second time stamp
    for line in proc.stdout:
        log.write(f"{time.time():.4f} {line}")


def identify_monitor():
    if shutil.which("tegrastats"): #Jetson
        return "tegrastats", ["tegrastats", "--interval", "100"]
    elif shutil.which("nvidia-smi"): #Desktop with nvidia GPU
                                        #TODO: change to TT gpu's hw_monitor
        return "nvidia_smi", ["nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu",
                "--format=csv,noheader", "-lms", "100"] #one csv line every 100 ms        
    return None

#starts the hardware monitor in the background, sets the hardware flags, and always cleans both up
@contextmanager
def hw_monitor(): 
    found_monitor = identify_monitor()     #chooses which hw monitor is used for hw runs
    if found_monitor is None:
        raise RuntimeError("No hardware monitor found")
    name, cmd = found_monitor
    
    config.SUBSTEP_TIMING = False
    config.HARDWARE_METRICS = True
    config.BENCH_DIR.mkdir(parents = True, exist_ok = True) #create benchmarking/ folder if missing
    log_path = config.BENCH_DIR / f"{name}.log" #tegrastats.log or nvidia_smi.log

    with open(log_path, "a") as log:
        #start suubprocess and send line right away to get accurate timestamps
        proc = subprocess.Popen(["stdbuf", "-oL"] + cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True)
        reader = threading.Thread(target = timestamp, args = (proc, log))
        reader.start()
        try:
            time.sleep(1) #let the monitor start
            yield name
        finally: #runs even if the pass crashes
            proc.terminate()
            proc.wait()
            reader.join() #wait for reader to write final line before closing log filke
            config.HARDWARE_METRICS = False

def print_logs(monitor, path):
    print(f"  UPDATED: {config.BENCH_DIR / f'{monitor}.log'}  ({monitor} readings, each line starts with time.time())")
    print(f"  UPDATED: {path}  (start/end time of each window, to pick out the readings inside it)")
########################################################################################################################################################################
#                         | | | |                                                                                                                                      #
#   Run Pipeline and Main V V V V                                                                                                                                      #
########################################################################################################################################################################

@torch.inference_mode()
def hardware_run_pipeline(model, img_paths, batch_size : int = config.BATCH_SIZE, device = config.DEVICE):
    with hw_monitor() as monitor:
        run.run_pipeline(model, img_paths, batch_size, device)

    results.store_hw_window(monitor, "full_pipeline", batch_size, run.hw_window["start"], run.hw_window["end"], config.STEP, config.OPTIMIZATION)
    print_logs(monitor, config.HARDWARE_PATH) #indicate which logs were updated and where they can be found

#test each substage in isolation over a fixed period and accumulates, averages, then stores associated hardware metrics
@torch.inference_mode()
def hardware_substages(model, images, batch_size = 1, device = config.DEVICE):
    #group the sample into batches of batch_size (same shape every call, so one compiled graph per stage)
    n_batches = max(1, config.HW_SAMPLE_SIZE // batch_size) #at least one batch even when batch_size > HW_SAMPLE_SIZE
    sample = images[: n_batches * batch_size]
    if len(sample) < batch_size:
        sys.exit(f"ERROR: NEED AT LEAST {batch_size} IMAGES FOR A BATCH {batch_size} STAGE RUN")
    batches = [sample[i:i + batch_size] for i in range(0, len(sample), batch_size)]

    ims, shapes, preds = [], [], []
    for b in batches: #prepare each stage's inputs before metrics are gathered
        im, shape = preprocess_stage(b, device)
        ims.append(im)
        shapes.append(shape)
        preds.append(inference_stage(model, im, device)[0]) #grabs the tensor portion of inference output
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()

    substages = {
        "idle": lambda i: time.sleep(0.01),
        "pre_processing": lambda i: preprocess_stage(batches[i], device),
        "inference": lambda i: inference_stage(model, ims[i], device),
        "post_processing": lambda i: postprocess_stage(preds[i], ims[i], shapes[i], config.CONF_THRES, config.IOU_THRESH, device),
    }

    #warm every substage up before the monitor start to avoid compile time overhead appearing inside the pre_processing and inference windows
    for substage_func in substages.values():
        for _ in range(config.COMPILE_WARMUP):
            substage_func(0)

    runs = []
    with hw_monitor() as monitor:
        for name, substage_func in substages.items(): #grab each substage from the lookup table
            time.sleep(config.COOLDOWN) #lets the chip cool back to idle temps before testing next substage, prevents previous run's tempts merging into this substage data
            if torch.device(device).type == "cuda": #force cpu to wait for gpu work before runs commence
                torch.cuda.synchronize()
            t_start = time.time()
            n = 0 #reset to only track iterations within a run
            while (time.time() - t_start) < config.PHASE_LIMIT: #run the substage under a time limit
                substage_func(n % len(batches)) #iterate through sample images
                n+=1 
            if torch.device(device).type == "cuda": #force cpu to wait for gpu work after runs finish
                torch.cuda.synchronize()
            t_end = time.time()
            runs.append({"monitor": monitor,"stage": name, "batch_size" : batch_size, "start" : t_start, "end" : t_end, "iterations" : n,
                         "step": config.STEP, "optimization": config.OPTIMIZATION}) #step and optimization added to  the end of the row
            per_batch = (t_end - t_start) / n * 1000 #ms per call (one batch)
            per_img = "" if name == "idle" else f", {per_batch / batch_size:.2f} ms per img" #idle processes no images
            print(f"{name:<16} {n} runs, {per_batch:.2f} ms per batch{per_img}")


    results.store_hw_stages(runs) #store captured metrics
    print_logs(monitor, config.HARDWARE_STAGES_PATH)
