import cv2 #open source computer vision libray utilized by ultralytics
import sys
import torch
import time
import torch_xla #TT CHANGE: sync(wait = False) in the e2e branch
import config
from timing import reset, _sync, _card_sync, signpost #signpost = tracy marker, same one timed() uses
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

@torch.no_grad() #no_grad run mode following available Tenstorrent demos and examples
                 #indicates not training, turn gradient tracking off
#based on benchmark class from ultralytics/engine/predictor.py
def run_pipeline(model, img_paths, batch_size: int = config.BATCH_SIZE, device = config.DEVICE, setting = ""):
    #reset async timigns from previous runs
    reset()

    images = load_images(img_paths)
    dropped = len(images) % batch_size
    if dropped: #drop impartial batches to avoid recompiling graphs due to unfamiliar input shapes
        print(f"WARNING: DROPPING LAST {dropped} IMAGES SO EVERY BATCH HAS {batch_size} IMAGES")
        images = images[:-dropped]
    if not images:
        sys.exit(f"ERROR: NEED AT LEAST {batch_size} READABLE IMAGES FOR BATCH SIZE {batch_size}")
    loaded = [images[i:i+batch_size] for i in range(0, len(images), batch_size)] #loads file paths present in each batch

    #warm up on every device for config.COMPILE_WARMUP iterations
    #torch.compile compiles on the first call whatever the device
    #second call recompiles part of the Detect head once
    for _ in range(config.COMPILE_WARMUP):
        im, shapes = preprocess_stage(loaded[0], device) #warmup with first loaded batch
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
    if not (setting == "e2e"):
        _card_sync() #finish warmup work before the hw window opens (_sync is off during hardware runs)
        hw_window["start"] = time.time()
        for imgs in loaded:
            _sync(device) #isolates work between stages like accelerator.synchronize does in ultralytics profile
            t0 = time.perf_counter()
            signpost("preprocess_start") #tracy marker for signifiying start of profiling preprocessing
            im, shapes = preprocess_stage(imgs, device)
            signpost("preprocess_end") #tracy marker for signifiying end of profiling preprocessing
            _sync(device)
            t1 = time.perf_counter()
            signpost("inference_start") #tracy marker for signifiying start of profiling inferncing
            predictions = inference_stage(model, im, device)
            signpost("inference_end") #tracy marker for signifiying end of profiling inferncing
            _sync(device)
            t2 = time.perf_counter()
            signpost("postprocess_start") #tracy marker for signifiying start of profiling post processing
            outputs.append(postprocess_stage(predictions, im, shapes, config.CONF_THRES, config.IOU_THRESH, device) );
            signpost("postprocess_end") #tracy marker for signifiying end of profiling postprocessing
            _sync(device)
            t3 = time.perf_counter()
            if not config.SUBSTEP_TIMING and not config.HARDWARE_METRICS:
                for s, dt in zip(config.STAGES, (t1-t0, t2-t1, t3-t2) ):
                    stage_wall_timings[s].append(dt * 1000)
            num_images += len(imgs) #keep track of number of images
        _card_sync() #wait for every queued batch before the hw window closes (outside the loop, once)
    elif setting == "e2e":
        batch_times = [] #end to end time of each batch, no added syncs between stages or batches
        _card_sync() #finish warmup work so it isn't billed to the first batch
        hw_window["start"] = time.time()
        signpost("timed_batch_loop_start") #only data inside this range is used for stage shares (excludes warmup)
        for imgs in loaded:
            t_start = time.perf_counter()
            signpost("preprocess_start")
            im, shapes = preprocess_stage(imgs, device)
            torch_xla.sync(wait = False) #send queued ops to the card now (no waiting) so they land inside this stage
            signpost("preprocess_end")
            signpost("inference_start")
            predictions = inference_stage(model, im, device)
            torch_xla.sync(wait = False)
            signpost("inference_end")
            signpost("postprocess_start")
            outputs.append(postprocess_stage(predictions, im, shapes, config.CONF_THRES, config.IOU_THRESH, device) );
            signpost("postprocess_end")
            t_end = time.perf_counter() #NOTE: postprocess reads boxes back to the host, so the batch's card work is finished here
            batch_times.append((t_end - t_start) * 1000)
            num_images += len(imgs)
        _card_sync() #finish card work
        signpost("timed_batch_loop_end")
    hw_window["end"] = time.time()

    #print results of runs
    if not config.HARDWARE_METRICS: #latency numbers aren't valid during hardware phase
        if setting == "e2e":
            csv_rows = results.display_e2e_results(batch_times, num_images, batch_size, device, hw_window["start"], hw_window["end"])
        else:
            csv_rows = results.display_results(stage_wall_timings, num_images, batch_size, device)
        if not config.PROFILE:  #profiled runs incurr nsys latency overhead, so shouldn't store results
            results.store_results(csv_rows)
    return outputs
