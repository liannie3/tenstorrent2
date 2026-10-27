# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            hardware_perf.py
# CREATED:         10/26/2026
# REVISION:        2.0
# ======================================= #

## tegrastats over the same sweep the other tests run. No stage timings -- the
## loop exists only to give the board something to do while tegrastats samples
## power, clocks, utilization and temperature.
##
## Writes tegrastats' own log and nothing else. Parsing is a separate job, so a
## bad parser cannot cost you the run.

import subprocess
import warnings
from pathlib import Path

import torch

from .. import data
from .. import model as model_lib

## 1000 ms is the tegrastats default, which gives ~8 samples for a bs=1
## sub-test. 100 ms is enough to see the clock ramp.
INTERVAL_MS = 100


## tegrastats behaves like a singleton -- a stale instance makes a new one print
## nothing. Stopping first is cheap and fixes that.
def _stop():
    try:
        subprocess.run(['tegrastats', '--stop'],
                       capture_output=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


## One sub-test: the whole image set at a single batch size.
## timings is unused -- the samples are the data, and they are in the log.
## Kept so every test has the same signature.
def run(config, encoder, processor, projector, paths, batch_size, timings):

    ## Loads parameters which are initialized in the jetson.yaml file
    device = config['run']['device']                               # config['run']['device'] returns cuda
    dtype = model_lib.resolve_dtype(config['run']['precision'])    # Goes to model.py and makes sure config dtype is allowed
    max_batches = config['run'].get('max_batches')                 # Sets the maximum batches
    images_seen = 0                                                # Image counter

    num_batches = (len(paths) + batch_size - 1) // batch_size      # Make sure to keep batches in power of 2 (1, 2, 4, 8, 16, ...)
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)                # Resolves if a batch size is like a 5 or an odd number not listed above

    ## One log per batch size, beside the other tests' output
    out_dir = Path(config['output']['results_dir']) / 'hardware_perf'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'tegrastats_bs{batch_size}.log'
    interval = config['run'].get('tegrastats_interval_ms', INTERVAL_MS)

    _stop()
    try:
        ## --logfile rather than a pipe, so nothing here has to drain stdout
        sampler = subprocess.Popen(['tegrastats', '--interval', str(interval),
                                    '--logfile', str(log_path)])
    except FileNotFoundError:
        warnings.warn('tegrastats not on PATH, so no hardware samples - run this on '
                      'the Orin', RuntimeWarning, stacklevel=2)
        sampler = None
    else:
        print(f'  tegrastats  : every {interval} ms -> {log_path}')

    try:
        with torch.inference_mode():
            for i, batch_paths in enumerate(data.iter_batches(paths, batch_size)):
                if i >= num_batches:
                    break

                ## Untimed on purpose -- tegrastats is the only instrument here
                images = data.load_images(batch_paths)                            # Return: Returns a list of the path and pixel data of images
                pixel_values = data.preprocess(processor, images, device, dtype)  # Return: Explained more in run_cpu.py
                outputs = encoder(pixel_values, output_hidden_states=True)
                features = model_lib.select_image_features(outputs, config)
                embeddings = projector(features)

                images_seen += len(batch_paths)

        ## Drains the queue so the last kernels land inside the sampled window
        torch.cuda.synchronize()
    finally:
        ## in a finally so a SystemExit mid-sweep does not leave it running
        if sampler is not None:
            _stop()
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()

    ## Shapes are the same every iteration, so the last batch describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen
