import cv2 #open source computer vision libray utilized by ultralytics
import sys
import torch
import time

import config
from timing import reset, _sync, process_timings
import results
from stages import preprocess_stage, inference_stage, postprocess_stage

hw_window = {} #store timestamp of the last run, used to line up hardware monitor logs

def load_images(img_paths):
    #read images before beginning timing as ultralyitics doesn't include this portion in preprocess timing
    images = []
    for path in img_paths:
        img = cv2.imread(str(path))
        if img is None: #cv2 returns None in case of error
            print(f"WARNING: SKIPPING UNREADABLE PATH: {path}")
            continue
        images.append(img)
    if not images:
        sys.exit("ERROR: NO IMAGES FROM DIRECTORY COULD BE READ")    
    return images

@torch.inference_mode() #indicates not training, for realistic performance
#based on benchmark class from ultralytics/engine/predictor.py
def run_pipeline(model, img_paths, batch_size: int = config.BATCH_SIZE, device = config.DEVICE, fp16 = config.FP16):
    #reset async timigns from previous runs
    reset()

    images = load_images(img_paths)
    loaded = [images[i:i+batch_size] for i in range(0, len(images), batch_size)] #loads file paths present in each batch

    #warmup model only on GPU
    if torch.device(device).type != "cpu": #NOTE: if intend to begin in steady state CPU for small img sample size, warmup CPU too (YOLOv8 doesn't do this)
        im, shapes = preprocess_stage(loaded[0], device, fp16) #warmup with first loaded batch
        predictions = inference_stage(model, im, device)
        postprocess_stage(predictions, im, shapes, config.CONF_THRES, config.IOU_THRESH, device)
        reset() #reset stored results from warmup
    
    #run model on all batches and store results 
    outputs =[]
    num_images = 0

    #NOTE: wall timing (sync between stages) measures real stage time ultralytics style;
    #      substep timing (async) gives extra context on GPU vs CPU time within stages.
    #      timing.sync() does nothing during substep or hardware runs
    #      NOTE: _sync does nothing when substep_timing is false
    stage_wall_timings = {s : [] for s in config.STAGES} #wall clock time to be hollistically measured like ultralytics does NOTE: only valid during sync runs
    hw_window["start"] = time.time()
    for imgs in loaded:
        _sync(device) #isolates work between stages like accelerator.synchronize does in ultralytics profile
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push("preprocess") #add nvtx marker for nsys and ncu analysis of preprocess 
        im, shapes = preprocess_stage(imgs, device, fp16)
        torch.cuda.nvtx.range_pop() #records CPU timestamp and removes the most recent nvtx marker (pop_back)
        _sync(device)
        t1 = time.perf_counter() 
        torch.cuda.nvtx.range_push("inference")
        predictions = inference_stage(model, im, device)
        torch.cuda.nvtx.range_pop()
        _sync(device)
        t2 = time.perf_counter()
        torch.cuda.nvtx.range_push("postprocess")
        outputs.append(postprocess_stage(predictions, im, shapes, config.CONF_THRES, config.IOU_THRESH, device) );
        torch.cuda.nvtx.range_pop()
        _sync(device)
        t3 = time.perf_counter()
        if not config.SUBSTEP_TIMING and not config.HARDWARE_METRICS:
            for s, dt in zip(config.STAGES, (t1-t0, t2-t1, t3-t2) ):
                stage_wall_timings[s].append(dt * 1000)
        num_images += len(imgs) #keep track of number of images 
    hw_window["end"] = time.time()
    process_timings()
    
    #print results of runs
    if not config.HARDWARE_METRICS: #latency numbers aren't valid during hardware phase
        csv_rows = results.display_results(stage_wall_timings, num_images, batch_size, device)
        results.store_results(csv_rows)
    return outputs
