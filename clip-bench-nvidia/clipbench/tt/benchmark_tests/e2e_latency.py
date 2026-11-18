# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            e2e_latency.py
# CREATED:         11/17/2026
# REVISION:        1.0
# ======================================= #

## Iteration = frame i from each view (FRONT, FRONT_LEFT, FRONT_RIGHT, etc)
## Same loop as the other tests, timed with perf_counter per frame and per sub-test.
## No tracy, no per-stage timing. Views run back to back, one synchronize per frame.

import time

import torch
import ttnn

from ...common import data
from .. import model_tt as model_lib
from ..tt_clip.device import get_device


## One sub-test: every frame, at a single view count.
def run(config, encoder, processor, projector, views_paths, views, timings):

    ## Loads parameters which are initialized in the tt.yaml file
    dtype = model_lib.resolve_dtype(config['run']['precision'])    # ttnn.bfloat16
    max_frames = config['run'].get('max_frames')                   # Caps the iterations
    images_seen = 0                                                # Image counter

    num_frames = data.num_frames(views_paths)                      # Iterations available, set by the shortest view
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    ## Start from an empty queue so warmup work isn't counted
    ttnn.synchronize_device(get_device())

    subtest_start = time.perf_counter_ns()                         # Subtest: Perf_Counter Start

    with torch.inference_mode():
        for i, frame_paths in enumerate(data.iter_frames(views_paths)):
            if i >= num_frames:
                break

            iter_start = time.perf_counter_ns()                    # Iter: Perf_Counter Start

            ## One sub-iteration per view, back to back, no synchronize
            for path in frame_paths:

                ## Decode + preprocess + upload [CPU]
                images = data.load_images([path])
                pixel_values = data.preprocess(processor, images, 'cpu', torch.float32)
                pixel_values = model_lib.to_device(pixel_values, dtype)

                ## Encode [TT]
                outputs = encoder.forward(pixel_values)
                features = model_lib.select_image_features(outputs, config)

                ## Project [TT]
                embeddings = projector(features)

                images_seen += 1

            ## Frame sync: wait for all views' device work to finish
            ttnn.synchronize_device(get_device())

            ## view -1 means all views: the whole rig for frame i, device work included
            iter_ms = (time.perf_counter_ns() - iter_start) / 1e6   # Iter: Perf_Counter End
            timings.append((views, 'iter', i, -1, iter_ms))

            if (i + 1) % 32 == 0 or i + 1 == num_frames:
                print(f'  frame       : {i + 1}/{num_frames}', flush=True)

    ## Already drained by the last frame sync; kept so 'subtest' always covers the device tail
    ttnn.synchronize_device(get_device())
    subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6     # Subtest: Perf_Counter End

    timings.append((views, 'subtest', 0, -1, subtest_ms))

    ## Shapes are the same every sub-iteration, so the last one describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))
        print(f'  throughput  : {1000 * images_seen / subtest_ms:.2f} images/s')

    return images_seen
