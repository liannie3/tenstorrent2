# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            cpu_pure_latency.py
# CREATED:         10/26/2026
# REVISION:        1.2
# ======================================= #

## Iteration = frame i from each view (FRONT, FRONT_LEFT, FRONT_RIGHT, etc)
## Within each iteration, one view runs at a time with no synchronization. This will allow us to see how the
## CPU and the GPU are working alongside one another

import time
import torch

from .. import data
from .. import model as model_lib


## One sub-test: every frame, at a single view count.
def run(config, encoder, processor, projector, views_paths, views, timings):

    ## Loads parameters which are initialized in the jetson.yaml file
    device = config['run']['device']                               # config['run']['device'] returns cuda
    dtype = model_lib.resolve_dtype(config['run']['precision'])    # Goes to model.py and makes sure config dtype is allowed
    max_frames = config['run'].get('max_frames')                   # Caps the iterations
    images_seen = 0                                                # Image counter

    num_frames = data.num_frames(views_paths)                      # Iterations available, set by the shortest view
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    subtest_start = time.perf_counter_ns()

    with torch.inference_mode():
        for i, frame_paths in enumerate(data.iter_frames(views_paths)):
            if i >= num_frames:
                break

            iter_start = time.perf_counter_ns()       # Iteration: Perf_Counter

            ## One sub-iteration per view, back to back. No synchronize here on
            ## purpose -- that is what lets the next view's host work overlap
            ## this view's GPU work.
            for view, path in enumerate(frame_paths):

                ## Decode [CPU]
                t0 = time.perf_counter_ns()               # Decode: Perf_Counter
                images = data.load_images([path])         # Return: Returns a list of the path and pixel data of images

                ## Preprocessing [CPU]
                t1 = time.perf_counter_ns()                                       # Decode & Preprocessing: Perf_counter
                pixel_values = data.preprocess(processor, images, device, dtype)  # Return: Explained more in run_cpu.py
                t2 = time.perf_counter_ns()                                       # Preprocessing & Encode: Perf_Counter

                ## Encode [CPU launch cost]
                outputs = encoder(pixel_values, output_hidden_states=True)
                features = model_lib.select_image_features(outputs, config)
                t3 = time.perf_counter_ns()                                       # Encode & Project: Perf_Counter

                ## Project [CPU launch cost]
                embeddings = projector(features)
                t4 = time.perf_counter_ns()                                       # Project: Perf_Counter

                ## Every stage is a host clock, so all four go straight into the
                ## buffer -- nothing has to wait for a synchronize here.
                timings.append((views, 'decode',     i, view, (t1 - t0) / 1e6))
                timings.append((views, 'preprocess', i, view, (t2 - t1) / 1e6))
                timings.append((views, 'encode',     i, view, (t3 - t2) / 1e6))
                timings.append((views, 'project',    i, view, (t4 - t3) / 1e6))

                ## This one view's four stages, host side
                timings.append((views, 'subiter', i, view, (t4 - t0) / 1e6))

                images_seen += 1

            ## view -1 means all views: the whole rig for frame i. Host time, so
            ## it ends when the last view was queued, not when it finished.
            iter_ms = (time.perf_counter_ns() - iter_start) / 1e6
            timings.append((views, 'iter', i, -1, iter_ms))

    ## Drains the queue so 'subtest' covers the GPU tail too, and so this
    ## sub-test's trailing kernels stay out of the next one's numbers.
    torch.cuda.synchronize()
    subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6               # Time for the entire subtest

    timings.append((views, 'subtest', 0, -1, subtest_ms))                     # Add the subtest times to the list

    ## Shapes are the same every sub-iteration, so the last one describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen
