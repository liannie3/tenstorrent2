import torch
import time
from contextlib import contextmanager                       
import shutil #detecting correct hardware monitor
import subprocess #execute tegrastats during hardware metric run

import config
import results
import run
from run import run_pipeline
from stages import preprocess_stage, inference_stage, postprocess_stage

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
        proc = subprocess.Popen(cmd, stdout = log, stderr = subprocess.STDOUT)
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

@torch.inference_mode()
def hardware_run_pipeline(model, img_paths, batch_size : int = config.BATCH_SIZE, device = config.DEVICE, fp16 = config.FP16):
    with hw_monitor() as monitor:
        run.run_pipeline(model, img_paths, batch_size, device, fp16)

    results.store_hw_window(monitor, "full_pipeline", run.hw_window["start"], run.hw_window["end"])

#test each substage in isolation over a fixed period and accumulates, averages, then stores associated hardware metrics
@torch.inference_mode()
def hardware_substages(model, images, device = config.DEVICE, fp16 = config.FP16):
    #prepare each stages inputs before metrics are gathered
    sample = images[:config.HW_SAMPLE_SIZE]
    ims, shapes = preprocess_stage(sample,device,fp16)
    preds = inference_stage(model, ims, device)[0] #grabs the tensor portion of inference output

    #lookup table to invoke each substage (and an idle for baseline) independently
    substages = {
        "idle": lambda i: time.sleep(0.01),
        "pre_processing": lambda i: preprocess_stage(sample[i:i+1], device, fp16),
        "inference": lambda i: inference_stage(model, ims[i:i+1], device),
        "post_processing": lambda i: postprocess_stage(preds[i:i+1], ims[i:i+1], shapes[i:i+1], config.CONF_THRES, config.IOU_THRESH, device),
    }

    runs = []
    with hw_monitor() as monitor:
        for name, substage_func in substages.items(): #grab each substage from the lookup tablew
            if torch.device(device).type == "cuda": #force cpu to wait for gpu work before runs commence
                torch.cuda.synchronize()
            t_start = time.time()
            n = 0 #reset to only track iterations within a run
            while (time.time() - t_start) < config.PHASE_LIMIT: #run the substage under a time limit
                substage_func(n % len(sample) ) #iterate through sample images
                n+=1 
            if torch.device(device).type == "cuda": #force cpu to wait for gpu work after runs finish
                torch.cuda.synchronize()
            t_end = time.time()
            runs.append({"monitor": monitor,"stage": name, "start" : t_start, "end" : t_end, "iterations" : n})
            print(f"{name:<16} {n} runs, {(t_end - t_start) / n * 1000:.2f} ms per run")
            time.sleep(1) #further discretize phases of substage testing

    results.store_hw_stages(runs)
