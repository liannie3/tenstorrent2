# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            cpu_pure_latency.py
# CREATED:         10/26/2026
# REVISION:        1.0
# ======================================= #

## perf_counter only, all four stages. Every number here is host time.
## decode and preprocess launch no kernels, so host time is their whole cost.
## encode and project are asynchronous, so host time is LAUNCH cost, not GPU
## time -- gpu_pure_latency.py measures those.

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

    num_batches = (len(paths) + batch_size - 1) // batch_size      # Make sure to keep batches in power of 2 (1, 2, 4, 8, 16, ...)
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)                # Resolves if a batch size is like a 5 or an odd number not listed above

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

            ## Encode [CPU launch cost]
            outputs = encoder(pixel_values, output_hidden_states=True)
            features = model_lib.select_image_features(outputs, config)
            t3 = time.perf_counter_ns()                                       # Encode & Project: Perf_Counter

            ## Project [CPU launch cost]
            embeddings = projector(features)
            t4 = time.perf_counter_ns()                                       # Project: Perf_Counter

            ## Every stage is a host clock, so all four go straight into the
            ## buffer -- nothing has to wait for a synchronize here.
            timings.append((batch_size, 'decode',     i, (t1 - t0) / 1e6))
            timings.append((batch_size, 'preprocess', i, (t2 - t1) / 1e6))
            timings.append((batch_size, 'encode',     i, (t3 - t2) / 1e6))
            timings.append((batch_size, 'project',    i, (t4 - t3) / 1e6))

            ## Sum of the four host stages. Unlike the GPU case this really is
            ## one interval, because the host runs them back to back.
            timings.append((batch_size, 'iter', i, (t4 - t0) / 1e6))

            images_seen += len(batch_paths)

    ## Drains the queue so 'subtest' covers the GPU tail too, and so this
    ## sub-test's trailing kernels stay out of the next one's numbers.
    torch.cuda.synchronize()
    subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6               # Time for the entire subtest

    timings.append((batch_size, 'subtest', 0, subtest_ms))                    # Add the subtest times to the list

    ## Shapes are the same every iteration, so the last batch describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen
