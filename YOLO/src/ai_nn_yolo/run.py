import sys
import time

import cv2 #open source computer vision libray utilized by ultralytics
import torch

import config
import results
from stages import inference_stage, postprocess_stage, preprocess_stage
from timing import _sync, reset, signpost
from tt_device import describe

hw_window = {} #store timestamp of the last run, used to line up hardware monitor logs (hardware step)


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


def make_batches(images, batch_size):
    #TT kernels are compiled for one batch size, so a trailing partial batch would trigger a fresh compile mid-run
    full = len(images) // batch_size
    if full == 0:
        sys.exit(f"ERROR: batch_size={batch_size} but only {len(images)} images were loaded")
    dropped = len(images) - full * batch_size
    if dropped:
        print(f"  NOTE: dropping the last {dropped} image(s) so every batch has exactly {batch_size}")
    return [images[i * batch_size : (i + 1) * batch_size] for i in range(full)]


def warmup(tt_model, ctx, batch, input_hw):
    #first call JIT-compiles every kernel (minutes on a cold cache, cached on disk after that) and prepares conv weights on
    #the card; the second call should already be at steady state
    #NOTE: the model is built for one batch size (every conv gets batch_size at construction, like the yolov8x demo)
    for _ in range(config.WARMUP_ITERS):
        im, shapes = preprocess_stage(batch, ctx, tt_model, input_hw)
        predictions = inference_stage(tt_model, im, ctx.device)
        postprocess_stage(predictions, shapes, ctx, input_hw, config.CONF_THRES, config.IOU_THRESH)
    ctx.sync()


@torch.inference_mode() #indicates not training, for realistic performance
#based on benchmark class from ultralytics/engine/predictor.py
def run_pipeline(tt_model, ctx, images, batch_size: int = config.BATCH_SIZE, input_hw=None, pass_name="wall_timing"):
    input_hw = tuple(input_hw or config.INPUT_HW)
    device = ctx.device
    reset() #reset timings from previous runs
    loaded = make_batches(images, batch_size)

    warmup(tt_model, ctx, loaded[0], input_hw)
    reset() #reset stored results from warmup

    #run model on all batches and store results
    outputs = []
    num_images = 0
    #NOTE: pass "wall_timing":    sync between stages, wall time per stage (ultralytics Profile style)
    #      pass "substep_timing": sync around every device substep, wall time per substep
    #      pass "e2e":            no syncs at all; the blocking device->host read in post-processing is the only wait,
    #                             so host dispatch of a batch overlaps with the card executing it (real loop latency)
    stage_wall_timings = {s: [] for s in config.STAGES} #only valid during the wall timing pass
    e2e_timings = []
    hw_window["start"] = time.time()
    for imgs in loaded:
        _sync(device) #isolates work between stages like accelerator.synchronize does in ultralytics profile
        t0 = time.perf_counter()
        signpost("preprocess")
        im, shapes = preprocess_stage(imgs, ctx, tt_model, input_hw)
        _sync(device)
        t1 = time.perf_counter()
        signpost("inference")
        predictions = inference_stage(tt_model, im, device)
        _sync(device)
        t2 = time.perf_counter()
        signpost("postprocess")
        outputs.append(postprocess_stage(predictions, shapes, ctx, input_hw, config.CONF_THRES, config.IOU_THRESH))
        _sync(device)
        t3 = time.perf_counter()
        if config.SYNC_STAGES and not config.SUBSTEP_TIMING:
            for s, dt in zip(config.STAGES, (t1 - t0, t2 - t1, t3 - t2)):
                stage_wall_timings[s].append(dt * 1000)
        e2e_timings.append((t3 - t0) * 1000)
        num_images += len(imgs) #keep track of number of images
    hw_window["end"] = time.time()

    #print results of runs
    meta = dict(ctx.settings(), input_hw=f"{input_hw[0]}x{input_hw[1]}", decode="device", device=describe(device))
    #NOTE: the whole detect head (DFL + box decode + sigmoid) runs on the card in the yolov8x-derived model
    csv_rows = results.display_results(pass_name, stage_wall_timings, e2e_timings, num_images, batch_size, meta)
    results.store_results(csv_rows)
    return outputs
