# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            deep_metrics.py
# CREATED:         10/25/2026
# REVISION:        2.0
# ======================================= #


import os
import subprocess
import sys
import warnings
from pathlib import Path

import torch
from torch.cuda import nvtx

from .. import config as cfg
from .. import data
from .. import model as model_lib
from ..reports import nsys as nsys_report
from ..reports.schema_nvidia import NSYS_ENV_GUARD, merge_nvtx_column, under_nsys


## Ranges are this test's only measurement, so no NVTX means nothing to record
def require_nvtx():
    try:
        nvtx.range_push('__probe__')
        nvtx.range_pop()
    except Exception as exc:
        raise RuntimeError('NVTX unavailable - run the latency test instead') from exc


## Keep in step with RANGE_PATTERN in clipbench/reports/nsys.py
def push(stage, batch_size):
    nvtx.range_push(f'{stage}/bs={batch_size}')


def pop():
    nvtx.range_pop()


## One sub-test: the whole image set at a single batch size.
## timings is unused -- the ranges are the data, and nsys holds them until the
## parent exports them. Kept so every test has the same signature.
def run(config, encoder, processor, projector, paths, batch_size, timings):
    require_nvtx()

    ## Loads parameters which are initialized in the jetson.yaml file
    device = config['run']['device']                               # config['run']['device'] returns cuda
    dtype = model_lib.resolve_dtype(config['run']['precision'])    # Goes to model.py and makes sure config dtype is allowed
    max_batches = config['run'].get('max_batches')                 # Sets the maximum batches
    images_seen = 0                                                # Image counter

    num_batches = (len(paths) + batch_size - 1) // batch_size      # Make sure to keep batches in power of 2 (1, 2, 4, 8, 16, ...)
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)                # Resolves if a batch size is like a 5 or an odd number not listed above


    push('subtest', batch_size)                       # Subtest: NVTX Push

    with torch.inference_mode():
        for i, batch_paths in enumerate(data.iter_batches(paths, batch_size)):
            if i >= num_batches:
                break

            push('iter', batch_size)                  # Iter: NVTX Push

            ## Decode [CPU]
            push('decode', batch_size)                # Decode: NVTX Push
            images = data.load_images(batch_paths)    # Return: Returns a list of the path and pixel data of images
            pop()                                     # Decode: NVTX Pop

            ## Preprocessing [CPU]
            push('preprocess', batch_size)                                    # Preprocessing: NVTX Push
            pixel_values = data.preprocess(processor, images, device, dtype)  # Return: Explained more in run_cpu.py
            pop()                                                             # Preprocessing: NVTX Pop

            ## Encode [GPU]
            push('encode', batch_size)                                        # Encode: NVTX Push
            outputs = encoder(pixel_values, output_hidden_states=True)
            features = model_lib.select_image_features(outputs, config)
            pop()                                                             # Encode: NVTX Pop

            ## Project [GPU]
            push('project', batch_size)               # Project: NVTX Push
            embeddings = projector(features)
            pop()                                     # Project: NVTX Pop

            pop()                                     # Iter: NVTX Pop

            images_seen += len(batch_paths)

    ## Ranges are host-side, so this is what makes 'subtest' enclose the GPU
    ## tail instead of ending when the last kernel was merely queued. Also
    ## keeps this sub-test's kernels out of the next one's ranges.
    torch.cuda.synchronize()
    pop()                                             # Subtest: NVTX Pop

    return images_seen


## True in the parent, False in the child nsys launched (the env guard)
def should_capture():
    return not under_nsys()


## Relaunch run_nvidia under nsys so the NVTX ranges get collected. child_args
## is what main() wants forwarded, e.g. ['--profile', 'jetson', '--test', 'nvtx'].
## Returns the .nsys-rep, or None if nsys is not installed.
def capture_under_nsys(config, child_args):

    ## Gets the results path which is defined in jetson.yaml
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    capture = results_dir / 'nsys_capture'                       # Makes the folder for the output nsys data

    #### IMPORTANT: DETERMINES THE FLAGS FOR NSYS PROFILE. SEE NSYS DOCS FOR MORE INFO ####
    ## nsys profile <flags> <command>. Everything from sys.executable on is the
    ## child nsys launches and traces; cwd=PROJECT_ROOT below is what makes
    ## -m clipbench.run_nvidia importable.
    cmd = ['nsys', 'profile',
           '--trace=cuda,nvtx',                           # kernels, memcpys, and our ranges -- nothing else
           '--sample=process-tree',
           '--cpuctxsw=process-tree',                     # need perf_event access we may not have
           '--force-overwrite', 'true',                   # else nsys exits nonzero on the second run
           '--output', str(capture),                      # base path; nsys appends .nsys-rep
           sys.executable, '-m', 'clipbench.run_nvidia',  # same interpreter and venv as the parent
           *child_args]

    env = dict(os.environ)                          # Builds the env the child process will run on
    env[NSYS_ENV_GUARD] = '1'                       # stops the child relaunching too

    ## Starts child process
    print('capturing NVTX ranges under nsys\n')
    try:
        result = subprocess.run(cmd, cwd=cfg.PROJECT_ROOT, env=env)
    except FileNotFoundError:
        warnings.warn('nsys is not on PATH, so there will be no nvtx_ms column. Install '
                      'Nsight Systems, or run the latency test instead.',
                      RuntimeWarning, stacklevel=2)
        return None

    ## Checks if nsys exited
    if result.returncode != 0:
        raise SystemExit(f'nsys profile exited {result.returncode}')
    return capture.with_suffix('.nsys-rep')


## Add the captured ranges to the stage CSVs the child just wrote
def merge_nvtx(rep_path, results_dir):

    try:
        csv_path = nsys_report.export_ranges(rep_path)
        timings = nsys_report.parse_ranges(csv_path)
    except (FileNotFoundError, RuntimeError) as exc:
        warnings.warn(f'capture kept at {rep_path}, but its ranges could not be read: '
                      f'{exc}', RuntimeWarning, stacklevel=2)
        return

    if not timings:
        warnings.warn(f'no NVTX ranges in {rep_path}', RuntimeWarning, stacklevel=2)
        return

    print('\nnvtx_ms merged into:')
    for stage, path in sorted(merge_nvtx_column(timings, results_dir).items()):
        print(f'{stage:>10}:', path)
    print('\ncapture:', rep_path)
