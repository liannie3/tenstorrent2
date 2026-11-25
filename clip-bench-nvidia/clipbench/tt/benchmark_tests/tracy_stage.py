# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            tracy_stage.py
# CREATED:         11/22/2026
# REVISION:        1.0
# ======================================= #

## Host vs device stage timing, run under the Tracy profiler. Ported ttnn model only (run.xla false).
##
##   python -m tracy -r -p -o analysis/results/tt/tracy -m clipbench.tt.run --test tracy_stage
##
## Iteration = frame i, every view stacked into ONE batch (views=6 -> [6,3,336,336]).
## Double-buffered so the host and the card can both stay busy:
##   upload frame i -> launch frame i (async) -> decode + preprocess frame i+1 -> sync
##
## Host:   perf_counter stages -> the usual CSVs (wait >> 0 = device-bound, wait ~ 0 = host-bound)
## Device: signposts -> tracy's ops_perf_results CSV + .tracy timeline (open in the Tracy GUI)

import time

import torch
import ttnn
from tracy import signpost

from ...common import data
from .. import model_tt as model_lib

MAX_FRAMES = 10         # Profiler buffers are small and profiling is slow, so keep this short


def mark(stage, views, i, edge):
    signpost(header=f'{stage}/v={views}/i={i}/{edge}')


## Decode + preprocess one frame, every view in one batch [CPU]
def prep(processor, frame_paths, views, i, timings):
    t0 = time.perf_counter_ns()
    images = data.load_images(list(frame_paths))
    t1 = time.perf_counter_ns()
    pixel_values = data.preprocess(processor, images, 'cpu', torch.float32)
    t2 = time.perf_counter_ns()
    timings.append((views, 'decode', i, -1, (t1 - t0) / 1e6))
    timings.append((views, 'preprocess', i, -1, (t2 - t1) / 1e6))
    return pixel_values


## One sub-test: every frame, at a single view count.
def run(config, encoder, processor, projector, views_paths, views, timings):
    if model_lib.XLA:
        raise SystemExit('tracy_stage needs the ported ttnn model -- set run.xla: false')

    dtype = model_lib.resolve_dtype(config['run']['precision'])    # ttnn.bfloat16
    device = model_lib.tt_device.get_device()
    frames = list(data.iter_frames(views_paths))[:MAX_FRAMES]

    ## Empty the profiler buffers of anything left from model load or the last sub-test
    model_lib.synchronize()
    ttnn.ReadDeviceProfiler(device)

    with torch.inference_mode():

        ## Warmup at this batch size (run.py's warmup is batch 1), then empty the profiler buffers
        warm = model_lib.to_device(prep(processor, frames[0], views, -1, []), dtype)
        projector(model_lib.select_image_features(encoder(warm, output_hidden_states=True), config))
        model_lib.synchronize()
        ttnn.ReadDeviceProfiler(device)

        pending = prep(processor, frames[0], views, 0, timings)                     # frame 0 has nothing to overlap

        for i in range(len(frames)):
            mark('frame', views, i, 'start')                                        # Frame: Signpost Start
            t0 = time.perf_counter_ns()

            ## Upload [CPU]
            pixel_values = model_lib.to_device(pending, dtype)
            t1 = time.perf_counter_ns()

            ## Encode + project [TT]. Returns once the ops are queued, not when the card is done
            mark('encode', views, i, 'start')                                       # Encode: Signpost Start
            outputs = encoder(pixel_values, output_hidden_states=True)
            features = model_lib.select_image_features(outputs, config)
            mark('encode', views, i, 'end')                                         # Encode: Signpost End
            mark('project', views, i, 'start')                                      # Project: Signpost Start
            embeddings = projector(features)
            mark('project', views, i, 'end')                                        # Project: Signpost End
            mark('frame', views, i, 'end')                                          # Frame: Signpost End
            t2 = time.perf_counter_ns()

            ## Decode + preprocess frame i+1 [CPU] while the card runs frame i
            if i + 1 < len(frames):
                pending = prep(processor, frames[i + 1], views, i + 1, timings)
            t3 = time.perf_counter_ns()

            ## Wait for frame i [host idle]
            model_lib.synchronize()
            t4 = time.perf_counter_ns()

            timings.append((views, 'upload', i, -1, (t1 - t0) / 1e6))
            timings.append((views, 'launch', i, -1, (t2 - t1) / 1e6))
            timings.append((views, 'wait', i, -1, (t4 - t3) / 1e6))
            timings.append((views, 'period', i, -1, (t4 - t0) / 1e6))

            ## Empty the profiler buffers before the next frame. After t4, so no host stage includes it
            ttnn.ReadDeviceProfiler(device)

    print('  pixel_values:', tuple(pixel_values.shape))
    print('  features    :', tuple(features.shape))
    print('  projected   :', tuple(embeddings.shape))

    return len(frames) * views
