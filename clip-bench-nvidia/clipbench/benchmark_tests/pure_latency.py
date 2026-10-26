# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            pure_latency.py
# CREATED:         10/25/2026
# REVISION:        2.0
# ======================================= #

import time
import torch

from .. import data
from .. import model as model_lib


## One sub-test: the whole image set at a single batch size.
def run(config, encoder, processor, projector, paths, batch_size, timings):

    ## Loads parameters which are initialized in the jetson.yaml file
    device = config['run']['device']                               # config['run']['device'] returns cuda
    dtype = model_lib.resolve_dtype(config['run']['precision'])    # Goes to model.py and makes sure config dtype is allowed
    max_batches = config['run'].get('max_batches')                 # Sets the maximum batches
    images_seen = 0                                                # Image counter
    batches_run = 0                                                # Batches counter

    num_batches = (len(paths) + batch_size - 1) // batch_size      # Make sure to keep batches in power of 2 (1, 2, 4, 8, 16, ...)
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)                # Resolves if a batch size is like a 5 or an odd number not listed above


    ## cudaEvents setup -- allocated before the loop so recording one costs
    ## nothing but the record itself
    encode_events = []
    project_events = []
    for i in range(num_batches):                                                # Loop over number of batches
        encode_events.append((torch.cuda.Event(enable_timing=True),             # Tuple of the start and end of encode events (t0, t1) B total (Number of batches)
                              torch.cuda.Event(enable_timing=True)))
        project_events.append((torch.cuda.Event(enable_timing=True),            # Tuple of the start and end of project events (t0, t1) B total (Number of batches)
                               torch.cuda.Event(enable_timing=True)))

    ## decode + preprocess per iteration, kept so the 'iter' row can be put
    ## together after the GPU numbers come back
    host_iter_ms = []

    subtest_start = time.perf_counter_ns()

    with torch.inference_mode():
        for i, batch_paths in enumerate(data.iter_batches(paths, batch_size)):
            if i >= num_batches:
                break

            ## Decode [CPU]
            t0 = time.perf_counter_ns()               # Decode: Perf_Counter
            images = data.load_images(batch_paths)    # Return: Returns a list of the path and pixel data of images

            ## Preprocessing [CPU]
            t1 = time.perf_counter_ns()                                       # Decode & Preprocessing: Perf_counter
            pixel_values = data.preprocess(processor, images, device, dtype)  # Return: Explained more in run_cpu.py
            t2 = time.perf_counter_ns()                                       # Preprocessing & Encode: Perf_Counter

            ## Encode [GPU]
            encode_events[i][0].record()                                      # Encode: Record starting time to cudaEvent
            outputs = encoder(pixel_values, output_hidden_states=True)
            features = model_lib.select_image_features(outputs, config)
            encode_events[i][1].record()                                      # Encode: Record ending time to cudaEvent

            ## Project [GPU]
            project_events[i][0].record()             # Project: Record starting time to cudaEvent
            embeddings = projector(features)
            project_events[i][1].record()             # Project: Record ending time to cudaEvent

            ## The host stages are known now and go straight into the buffer.
            ## The GPU stages cannot: elapsed_time() on an event that has not
            ## happened yet would block, which is the stall this file exists
            ## to avoid. They are read after the synchronize below.
            timings.append((batch_size, 'decode',     i, (t1 - t0) / 1e6))
            timings.append((batch_size, 'preprocess', i, (t2 - t1) / 1e6))
            host_iter_ms.append((t2 - t0) / 1e6)

            images_seen += len(batch_paths)
            batches_run += 1

    torch.cuda.synchronize()
    subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6               # Time for the entire subtest

    timings.append((batch_size, 'subtest', 0, subtest_ms))                    # Add the subtest times to the list

    ## The stream is drained, so every event has completed and these reads
    ## return immediately. elapsed_time() gives milliseconds as a float.
    for i in range(batches_run):
        encode_ms = encode_events[i][0].elapsed_time(encode_events[i][1])      # Finds the elapsed time for each of the events (float)
        project_ms = project_events[i][0].elapsed_time(project_events[i][1])
        timings.append((batch_size, 'encode',  i, encode_ms))                  # Appends them to the timing buffer
        timings.append((batch_size, 'project', i, project_ms))

        timings.append((batch_size, 'iter', i,
                        host_iter_ms[i] + encode_ms + project_ms))

    ## Shapes are the same every iteration, so the last batch describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen
