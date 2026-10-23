# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            run_nvidia.py
# CREATED:         10/19/2026
# REVISION:        2.0
# ======================================= #

"""NVIDIA runner. Times the host stages with perf_counter and the GPU stages
with CUDA events, and writes the same per-stage CSVs run_cpu.py writes.

    sudo nvpmodel -m 0        # MAXN -- do this before benchmarking
    sudo jetson_clocks        # pin clocks so sub-tests are comparable

    python -m clipbench.run_nvidia --profile jetson

Why not perf_counter for every stage like run_cpu.py: a CUDA call returns as
soon as its kernels are queued, so a host clock around `encoder(...)` measures
how long it took to *launch* the work, not how long the GPU spent on it. The
usual fix is torch.cuda.synchronize() after each stage, which drains the queue
four times an iteration and destroys exactly the CPU/GPU overlap the Orin
relies on in deployment.

A CUDA event is a timestamp written into the stream instead. Recording one does
not make the host wait, so the events sit around encode and project without
breaking the overlap, and the durations are read back after the sub-test's one
synchronize -- by which point every event has completed, so the read does not
block either. Host-bound stages (decode, preprocess) stay on perf_counter
because they do no GPU work for an event to time.

The NVTX ranges carry no measurements -- they are inert until a profiler
collects them. So by default this runner relaunches itself once under
`nsys profile` and merges the ranges into the stage CSVs as an nvtx_ms column,
leaving the .nsys-rep beside them for the Nsight Systems GUI. --no-nsys skips
all of that and runs in-process.

    python -m clipbench.run_nvidia --profile jetson --no-nsys

A profiled run carries CUPTI overhead on every kernel launch, so use --no-nsys
for headline numbers.

Range names carry the batch size but deliberately NOT the iteration index.
`nsys stats --report nvtx_sum` groups by name, so 'encode/bs=4' stays one row
across every iteration of that sub-test; putting the index in the name would
make every range unique and defeat the aggregation.
"""

import argparse
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.cuda import nvtx

from . import config as cfg
from . import data
from . import model as model_lib
from .reports import nsys as nsys_report
from .reports.schema_nvidia import (NSYS_ENV_GUARD, merge_nvtx_column,
                                    under_nsys, write_manifest, write_stage_csvs)


## Function checks to make sure NVTX is available
def _nvtx_available():
    try:
        nvtx.range_push('__probe__')
        nvtx.range_pop()
        return True
    except Exception:
        return False


NVTX_AVAILABLE = _nvtx_available()  # First check if NVTX is available


## The name format lives here only -- keep it in step with RANGE_PATTERN in
## clipbench/reports/nsys.py, which is what parses it back out of the capture
def push(stage, batch_size):
    if NVTX_AVAILABLE:
        nvtx.range_push(f'{stage}/bs={batch_size}')


def pop():
    if NVTX_AVAILABLE:
        nvtx.range_pop()


## One sub-test: the whole image set at a single batch size.
def run_subtest(config, encoder, processor, projector, paths, batch_size, timings):

    ## Loads parameters which are initialized in the cpu.yaml file
    device = config['run']['device']                               # config['run']['device'] returns cuda    
    dtype = model_lib.resolve_dtype(config['run']['precision'])    # Goes to model.py and makes sure config dtype is allowed
    max_batches = config['run'].get('max_batches')                 # Sets the maximum batches
    images_seen = 0                                                # Image counter
    batches_run = 0                                                # Batches counter

    ## To use cudaEvents, we must set the flags prior to running the program
    ## Set before the loop starts
    num_batches = (len(paths) + batch_size - 1) // batch_size                   # Make sure to keep batches in power of 2 (1, 2, 4, 8, 16, ...)
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)                             # Resolves if a batch size is like a 5 or and odd number not listed above

    ## cudaEvents setup
    encode_events = []
    project_events = []
    for i in range(num_batches):                                                # Loop over number of batches
        encode_events.append((torch.cuda.Event(enable_timing=True),             # Tuple of the start and end of encode events (t0, t1) B total (Number of batches)
                              torch.cuda.Event(enable_timing=True)))            
        project_events.append((torch.cuda.Event(enable_timing=True),            # Tuple of the start and end of decode events (t0, t1) B total (Number of batches)
                               torch.cuda.Event(enable_timing=True)))

    ## decode + preprocess per iteration, kept so the 'iter' row can be put
    ## together after the GPU numbers come back
    host_iter_ms = []

    push('subtest', batch_size)                       # Subtest: NVTX Push
    subtest_start = time.perf_counter_ns()

    ## i is the batch index and the index into the event pairs above; the break
    ## enforces max_batches, which iter_batches knows nothing about.
    ## 100 images at bs=8 -> 13 batches, so i runs 0..12 (last batch has 4).
    with torch.inference_mode():
        for i, batch_paths in enumerate(data.iter_batches(paths, batch_size)):
            if i >= num_batches:
                break

            push('iter', batch_size)                  # Iter: NVTX Push

            ## Decode [CPU]
            t0 = time.perf_counter_ns()               # Decode: Perf_Counter
            push('decode', batch_size)                # Decode: NVTX Push
            images = data.load_images(batch_paths)    # Return: Returns a list of the path and pixel data of images
            pop()                                     # Decode: NVTX Pop

            ## Preprocessing [CPU]
            t1 = time.perf_counter_ns()                                       # Decode & Preprocessing: Perf_counter
            push('preprocess', batch_size)                                    # Preprocessing: NVTX Push
            pixel_values = data.preprocess(processor, images, device, dtype)  # Return: Explained more in run_cpu.py
            pop()                                                             # Preprocessing: NVTX Pop
            t2 = time.perf_counter_ns()                                       # Preprocessing & Encode: Perf_Counter

            ## Encode [GPU]
            push('encode', batch_size)                                        # Encode: NVTX Push
            encode_events[i][0].record()                                      # Encode: Record starting time to cudaEvent
            outputs = encoder(pixel_values, output_hidden_states=True)        
            features = model_lib.select_image_features(outputs, config)
            encode_events[i][1].record()                                      # Encode: Record ending time to cudaEvent
            pop()                                                             # Encode: NVTX Pop

            push('project', batch_size)               # Project: NVTX Push
            project_events[i][0].record()             # Project: Record starting time to cudaEvent
            embeddings = projector(features)
            project_events[i][1].record()             # Project: Record ending time to cudaEvent
            pop()                                     # Project: NVTX Pop

            pop()       # Iter: NVTX Pop

            ## The host stages are known now and go straight into the buffer.
            ## The GPU stages cannot: elapsed_time() on an event that has not
            ## happened yet would block, which is the stall this file exists
            ## to avoid. They are read after the synchronize below.
            timings.append((batch_size, 'decode',     i, (t1 - t0) / 1e6))
            timings.append((batch_size, 'preprocess', i, (t2 - t1) / 1e6))
            host_iter_ms.append((t2 - t0) / 1e6)

            images_seen += len(batch_paths)
            batches_run += 1

    ## The only synchronize in the timed region -- once per batch size, not per
    ## stage, so overlap survives. Draining here pulls the GPU tail into
    ## 'subtest' (the wall-clock envelope throughput uses) and keeps this
    ## sub-test's trailing kernels out of the next one's numbers.
    torch.cuda.synchronize()
    subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6               # Time for the entire subtest     
    pop()               # Subtest: NVTX Pop

    timings.append((batch_size, 'subtest', 0, subtest_ms))                    # Add the subtest times to the list

    ## The stream is drained, so every event has completed and these reads
    ## return immediately. elapsed_time() gives milliseconds as a float.
    for i in range(batches_run):
        encode_ms = encode_events[i][0].elapsed_time(encode_events[i][1])      # Finds the elapsed time for each of the events (float)
        project_ms = project_events[i][0].elapsed_time(project_events[i][1])
        timings.append((batch_size, 'encode',  i, encode_ms))                  # Appends them t the timing buffer
        timings.append((batch_size, 'project', i, project_ms))

        ## 'iter' here is the sum of the four stages, not a wall-clock reading.
        ## One iteration's host stages overlap the previous iteration's GPU
        ## stages, so there is no single interval that contains all four and
        ## nothing else. Use 'subtest' for throughput; this column is for
        ## comparing the stage mix against run_cpu.py.
        timings.append((batch_size, 'iter', i,
                        host_iter_ms[i] + encode_ms + project_ms))

    ## Shapes are the same every iteration, so the last batch describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen


## Relaunch this module under nsys so the NVTX ranges get collected. Returns
## the .nsys-rep, or None if nsys is not installed.
def _capture_under_nsys(config, profile):

    ## Gets the results path which is defined in jetson.yaml
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    capture = results_dir / 'nsys_capture'                       # Makes the a folder for the output nsys data

    #### IMPORTANT: DETERMINES THE FLAGS FOR NSYS PROFILE. SEE NSYS DOCS FOR MORE INFO ####
    ## nsys profile <flags> <command>. Everything from sys.executable on is the
    ## child nsys launches and traces; cwd=PROJECT_ROOT below is what makes
    ## -m clipbench.run_nvidia importable.
    cmd = ['nsys', 'profile',
           '--trace=cuda,nvtx',                           # kernels, memcpys, and our ranges -- nothing else
           '--sample=process-tree', 
           '--cpuctxsw=process-tree',            # need perf_event access we may not have
           '--force-overwrite', 'true',                   # else nsys exits nonzero on the second run
           '--output', str(capture),                      # base path; nsys appends .nsys-rep
           sys.executable, '-m', 'clipbench.run_nvidia',  # same interpreter and venv as the parent
           '--profile', profile]                          # --nsys not forwarded; the env guard stops the relaunch

    env = dict(os.environ)                          # Builds the OS the child process will run on
    env[NSYS_ENV_GUARD] = '1'                       # stops the child relaunching too

    ## Starts child terminal
    print('capturing NVTX ranges under nsys\n')
    try:
        result = subprocess.run(cmd, cwd=cfg.PROJECT_ROOT, env=env)      
    except FileNotFoundError:
        warnings.warn('nsys is not on PATH, so there will be no nvtx_ms column. Install '
                      'Nsight Systems, or pass --no-nsys to stop trying.',
                      RuntimeWarning, stacklevel=2)
        return None

    ## Checks if nsys exited
    if result.returncode != 0:
        raise SystemExit(f'nsys profile exited {result.returncode}')
    return capture.with_suffix('.nsys-rep')


## Add the captured ranges to the stage CSVs the child just wrote
def _merge_nvtx(rep_path, results_dir):

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


def main(argv=None):
    ## Reads from command line (important to have --profile when running program)
    parser = argparse.ArgumentParser(description='CLIP encoder + MLP projector sweep, CUDA event timing')
    parser.add_argument('--profile', default='jetson', choices=cfg.available_profiles(),
                        help='config profile from configs/ (default: jetson)')
    parser.add_argument('--nsys', action=argparse.BooleanOptionalAction, default=True,
                        help='collect NVTX ranges under nsys (default: on)')
    args = parser.parse_args(argv)

    ## 1. Config
    config = cfg.load_config(args.profile)
    run = config['run']

    ## Make sure the run is on an nvidia device
    if run['target'] != 'nvidia':
        raise SystemExit(
            f"profile {args.profile!r} has run.target {run['target']!r} - run it with:\n"
            f'  python -m clipbench.run_cpu --profile {args.profile}'
        )

    ## Checks if CUDA is available
    if not torch.cuda.is_available():
        raise SystemExit(
            'run_nvidia.py times the GPU stages with CUDA events and needs a working '
            'CUDA device, but torch.cuda.is_available() is False.\n'
            'For a CPU-only run use:\n'
            '  python -m clipbench.run_cpu --profile cpu'
        )

    ## Checks if NVTX is available
    if not NVTX_AVAILABLE:
        warnings.warn(
            'NVTX is unavailable, so the ranges will be no-ops and an nsys capture of '
            'this run will contain no stage annotations. The CSV timings are unaffected.',
            RuntimeWarning,
            stacklevel=2,
        )

    ## This is where the Parent OS Process gets blocked so the Child OS Process can run the 
    ## program.
    ## If nsys is being ran with the program, there may be added latency
    if args.nsys and NVTX_AVAILABLE and not under_nsys():
        rep_path = _capture_under_nsys(config, args.profile)
        if rep_path is not None:
            _merge_nvtx(rep_path, Path(config['output']['results_dir']))
            return 0
        print('running in-process instead\n')

    print(f"profile: {config['profile']}  device: {run['device']}  "
          f"precision: {run['precision']}")

    ## 2. Model pieces, built once and reused by every sub-test
    processor = model_lib.load_processor(config)                                 # from model.py
    encoder = model_lib.load_encoder(config)                                    # from model.py
    projector = model_lib.load_projector(config, encoder.config.hidden_size)    # from model.py

    ## 3. The image list, identical for every sub-test
    paths = data.load_paths(config)                                             # list of image paths
    print(f"found {len(paths)} images in {config['data']['path']}")
    if not paths:
        raise SystemExit('no images found - check data.path in configs/base.yaml')

    ## 4. The timing buffer. Appended to all through the sweep below, read once
    ## at step 6 and not before.
    timings = []        # (batch_size, stage, iter, duration_ms)

    ## 5. One sub-test per batch size
    images_by_batch_size = {}
    for batch_size in run['batch_sizes']:
        print(f'\n=== sub-test: batch size {batch_size} over {len(paths)} images ===')
        images_seen = run_subtest(
            config, encoder, processor, projector, paths, batch_size, timings)
        images_by_batch_size[batch_size] = images_seen
        print(f'  images      : {images_seen}')

    ## 6. Sweep is over, so now the buffer gets read and the files get written.
    ## The manifest records how the run was configured -- including the Jetson
    ## power mode, since DVFS makes two runs in different modes incomparable.
    results_dir = Path(config['output']['results_dir'])
    print('\nmanifest:', write_manifest(config, images_by_batch_size,
                                        results_dir / 'manifest.json'))
    for stage, path in write_stage_csvs(timings, results_dir).items():
        print(f'{stage:>10}:', path)

    return 0


## Footnote on preprocess: data.preprocess ends in a .to(device) on a pageable
## tensor, and CUDA synchronizes the stream before a pageable H2D copy. So the
## preprocess number includes waiting for the previous iteration's encode to
## finish. Switching that copy to pinned memory with non_blocking=True would
## remove the wait and give a cleaner split -- but then preprocess becomes
## asynchronous too and needs its own event pair, like encode and project.
if __name__ == '__main__':
    raise SystemExit(main())
