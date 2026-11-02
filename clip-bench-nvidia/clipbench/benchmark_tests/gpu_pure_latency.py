# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            gpu_pure_latency.py
# CREATED:         10/26/2026
# REVISION:        2.0
# ======================================= #


## Iteration = frame i from each view (FRONT, FRONT_LEFT, FRONT_RIGHT, etc)
## Within each iteration, one view runs at a time with no synchronization. This will allow us to see how the
## CPU and the GPU are working alongside one another

import time
import torch

from .. import data
from .. import model as model_lib


## Start/end pair, allocated before the loop so recording one costs nothing but
## the record itself
def _event_pair():
    return (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))


## One sub-test: every frame, at a single view count.
def run(config, encoder, processor, projector, views_paths, views, timings):

    ## Loads parameters which are initialized in the jetson.yaml file
    device = config['run']['device']                               # config['run']['device'] returns cuda
    dtype = model_lib.resolve_dtype(config['run']['precision'])    # Goes to model.py and makes sure config dtype is allowed
    max_frames = config['run'].get('max_frames')                   # Caps the iterations
    images_seen = 0                                                # Image counter
    frames_run = 0                                                 # Iteration counter

    num_frames = data.num_frames(views_paths)                      # Iterations available, set by the shortest view
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    ## cudaEvents setup -- one pair per (frame, view) for the stages, plus one
    ## pair per frame for the iteration envelope
    encode_events = [[_event_pair() for _ in range(views)] for _ in range(num_frames)]
    project_events = [[_event_pair() for _ in range(views)] for _ in range(num_frames)]
    iter_events = [_event_pair() for _ in range(num_frames)]

    ## perf_counter appears once, for the sub-test envelope only -- throughput
    ## needs a wall clock and no event pair can give one
    subtest_start = time.perf_counter_ns()

    with torch.inference_mode():
        for i, frame_paths in enumerate(data.iter_frames(views_paths)):
            if i >= num_frames:
                break

            iter_events[i][0].record()                # Iteration: Record starting time to cudaEvent

            for view, path in enumerate(frame_paths):

                ## Decode + preprocess [CPU, untimed here]
                images = data.load_images([path])                                  # Return: Returns a list of the path and pixel data of images
                pixel_values = data.preprocess(processor, images, device, dtype)   # Return: Explained more in run_cpu.py

                ## Encode [GPU]
                encode_events[i][view][0].record()                                # Encode: Record starting time to cudaEvent
                outputs = encoder(pixel_values, output_hidden_states=True)
                features = model_lib.select_image_features(outputs, config)
                encode_events[i][view][1].record()                                # Encode: Record ending time to cudaEvent

                ## Project [GPU]
                project_events[i][view][0].record()       # Project: Record starting time to cudaEvent
                embeddings = projector(features)
                project_events[i][view][1].record()       # Project: Record ending time to cudaEvent

                images_seen += 1

            iter_events[i][1].record()                # Iteration: Record ending time to cudaEvent
            frames_run += 1

    ## The only synchronize in the timed region -- once per sub-test, not per
    ## stage or per view, so overlap survives. Draining here pulls the GPU tail
    ## into 'subtest' and keeps this sub-test's trailing kernels out of the next
    ## one's numbers.
    torch.cuda.synchronize()
    subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6               # Time for the entire subtest

    timings.append((views, 'subtest', 0, -1, subtest_ms))                     # Add the subtest times to the list

    ## The stream is drained, so every event has completed and these reads
    ## return immediately. elapsed_time() gives milliseconds as a float.
    for i in range(frames_run):
        for view in range(views):
            encode_ms = encode_events[i][view][0].elapsed_time(encode_events[i][view][1])     # Finds the elapsed time for each of the events (float)
            project_ms = project_events[i][view][0].elapsed_time(project_events[i][view][1])
            timings.append((views, 'encode',  i, view, encode_ms))                            # Appends them to the timing buffer
            timings.append((views, 'project', i, view, project_ms))

            ## GPU work for this one view
            timings.append((views, 'subiter', i, view, encode_ms + project_ms))

        ## view -1 means all views: GPU span of the whole rig for frame i,
        ## bubbles included, so it is >= the sum of the sub-iterations above
        timings.append((views, 'iter', i, -1,
                        iter_events[i][0].elapsed_time(iter_events[i][1])))

    ## Shapes are the same every sub-iteration, so the last one describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen
