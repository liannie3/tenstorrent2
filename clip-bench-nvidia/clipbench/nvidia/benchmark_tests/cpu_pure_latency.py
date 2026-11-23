# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            cpu_pure_latency.py
# CREATED:         10/26/2026
# REVISION:        1.2
# ======================================= #

## Iteration = frame i from each view (FRONT, FRONT_LEFT, FRONT_RIGHT, etc)
## Within each iteration, one view runs at a time with no synchronization. This will allow us to see how the
## CPU and the GPU are working alongside one another. One synchronize after the last view of each frame.

import time
import torch

from ...common import data
from .. import model_nvidia as model_lib


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

    ## inference_mode wraps the whole sub-test because nothing here ever trains
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
                t0 = time.perf_counter_ns()                                       # Decode: Perf_Counter
                images = data.load_images([path])                                 # Return: Returns a list of the path and pixel data of images

                ## Preprocessing [CPU]                                            # Time Marker: End Load Images, Start Preprocessing                             (900, 1600, 3) uint8              
                t1 = time.perf_counter_ns()                                       # Return: Resize: Converts 1600x900px res image to 597x336px                    (336, 597, 3) uint8               
                pixel_values = data.preprocess(processor, images, device, dtype)  #         Center: Crops to 336x336px                                            (336, 336, 3) uint8               
                t2 = time.perf_counter_ns()                                       #         Rescale: Converts uint8 integers to floats [0, 255] --> [0.0, 1.0]    (336, 336, 3) float32             
                                                                                  #         Normalize: Each R,G,B gets a different mean and STD                   (336, 336, 3) float32             
                                                                                  #         Stack: Stacks the images in the NCHW tensor                           (1, 3, 336, 336) float32          
                                                                                  #         To: Moves to device and casts precision (fp32 cpu / fp16 jetson)      (1, 3, 336, 336) float32 / float16

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

            ## Frame sync: wait for all views' GPU work to finish
            torch.cuda.synchronize()

            ## view -1 means all views: the whole rig for frame i, GPU work included
            iter_ms = (time.perf_counter_ns() - iter_start) / 1e6
            timings.append((views, 'iter', i, -1, iter_ms))

    ## Already drained by the last frame sync; kept so 'subtest' always covers the GPU tail
    torch.cuda.synchronize()
    subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6               # Time for the entire subtest

    timings.append((views, 'subtest', 0, -1, subtest_ms))                     # Add the subtest times to the list

    ## Shapes are the same every sub-iteration, so the last one describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen
