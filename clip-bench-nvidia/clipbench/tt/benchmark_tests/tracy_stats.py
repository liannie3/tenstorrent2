# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            tracy_stats.py
# CREATED:         11/17/2026
# REVISION:        1.0
# ======================================= #

"""Per-stage device latency from the Tracy device profiler. TT's version of nvidia/deep_metrics.py.

Signposts mark where each stage starts and ends in the op stream. Every device op
queued between a stage's start and end signposts gets counted toward that stage,
and its kernel time is measured on the card. No synchronize is needed for that, so
the loop keeps the same no-sync overlap as the NVIDIA tests.

The signposts only show up when the process runs under `python -m tracy -r`, so
capture_under_tracy() relaunches the runner that way, and parse_ops_report() turns
the ops_perf_results CSV into the usual (views, stage, iter, view, ms) rows.
"""

import csv
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import torch
import ttnn

from ...common import config as cfg
from ...common import data
from .. import model_tt as model_lib
from ..tt_clip.device import get_device

## Set on the child that runs under `python -m tracy`
TRACY_ENV_GUARD = 'CLIPBENCH_UNDER_TRACY'

## Must stay in step with mark() below
SIGNPOST_PATTERN = re.compile(
    r'^(?P<stage>\w+)/v=(?P<views>\d+)/i=(?P<iter>\d+)/view=(?P<view>-?\d+)/(?P<edge>start|end)$')

## Column names in tt-metal's ops_perf_results CSV
OP_TYPE = 'OP TYPE'
OP_CODE = 'OP CODE'
KERNEL_NS = 'DEVICE KERNEL DURATION [ns]'


def under_tracy():
    return os.environ.get(TRACY_ENV_GUARD) == '1'


## Signposts are the only thing this test records, so fail early if tracy is missing
def require_tracy():
    try:
        from tracy import signpost  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('tracy module not found - build tt-metal with ENABLE_TRACY=ON') from exc


## view -1 means all views, for ranges that span the whole rig
def mark(stage, views, i, view, edge):
    from tracy import signpost
    signpost(header=f'{stage}/v={views}/i={i}/view={view}/{edge}')


## Copies the device profiler buffers to the host before they fill up. Syncs the device.
def read_profiler():
    ttnn.ReadDeviceProfiler(get_device())


## One sub-test: every frame, at a single view count.
## timings is unused -- the signposts are the data. Kept so every test has the same signature.
def run(config, encoder, processor, projector, views_paths, views, timings):
    require_tracy()

    ## Loads parameters which are initialized in the tt.yaml file
    dtype = model_lib.resolve_dtype(config['run']['precision'])    # ttnn.bfloat16
    max_frames = config['run'].get('tracy_max_frames')             # Profiling is slow, so tt.yaml caps it separately
    if max_frames is None:
        max_frames = config['run'].get('max_frames')
    images_seen = 0                                                # Image counter

    num_frames = data.num_frames(views_paths)                      # Iterations available, set by the shortest view
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    ## Flush the warmup's ops so they don't take up buffer space
    read_profiler()
    print(f'  profiling   : {num_frames} frames', flush=True)

    mark('subtest', views, 0, -1, 'start')                         # Subtest: Signpost Start

    with torch.inference_mode():
        for i, frame_paths in enumerate(data.iter_frames(views_paths)):
            if i >= num_frames:
                break

            mark('iter', views, i, -1, 'start')                    # Iter: Signpost Start

            ## One sub-iteration per view, no synchronize between them
            for view, path in enumerate(frame_paths):

                mark('subiter', views, i, view, 'start')           # Subiter: Signpost Start

                ## Decode + preprocess + upload [CPU, no device ops so not timed here]
                images = data.load_images([path])
                pixel_values = data.preprocess(processor, images, 'cpu', torch.float32)
                pixel_values = model_lib.to_device(pixel_values, dtype)

                ## Encode [TT]
                mark('encode', views, i, view, 'start')            # Encode: Signpost Start
                outputs = encoder.forward(pixel_values)
                features = model_lib.select_image_features(outputs, config)
                mark('encode', views, i, view, 'end')              # Encode: Signpost End

                ## Project [TT]
                mark('project', views, i, view, 'start')           # Project: Signpost Start
                embeddings = projector(features)
                mark('project', views, i, view, 'end')             # Project: Signpost End

                mark('subiter', views, i, view, 'end')             # Subiter: Signpost End

                images_seen += 1

            mark('iter', views, i, -1, 'end')                      # Iter: Signpost End

            ## Frame sync: wait for all views' device work to finish
            ttnn.synchronize_device(get_device())

            ## One frame of ViT-L is thousands of ops, so the buffers are read every frame.
            ## Outside every stage range, and device kernel times don't see host waits.
            read_profiler()

            if (i + 1) % 5 == 0 or i + 1 == num_frames:
                print(f'  frame       : {i + 1}/{num_frames}', flush=True)

    mark('subtest', views, 0, -1, 'end')                           # Subtest: Signpost End
    read_profiler()

    ## Shapes are the same every sub-iteration, so the last one describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen


## Relaunch tt/run.py under tracy so the signposts and device ops get collected.
## child_args is what main() wants forwarded, e.g. ['--profile', 'tt'].
## Returns the ops_perf_results CSV, or None if tracy failed or wrote none.
def capture_under_tracy(config, child_args):
    capture = Path(config['output']['results_dir']) / 'tracy_capture'
    capture.mkdir(parents=True, exist_ok=True)

    ## python -m tracy <flags> -m <module> <args>
    cmd = [sys.executable, '-m', 'tracy',
           '-r',                                     # generate the ops report after the run
           '-p',                                     # only profile enabled zones, less overhead
           '-v',                                     # verbose
           '-o', str(capture)]                       # report folder

    ## Pip/conda ttnn has no capture/csvexport tools, so point at a prebuilt copy
    tools = config['run'].get('tracy_tools_folder')
    if tools:
        cmd += ['--tracy-tools-folder', tools]

    cmd += ['-m', 'clipbench.tt.run', *child_args]

    env = dict(os.environ)
    env[TRACY_ENV_GUARD] = '1'                       # stops the child relaunching too
    env.setdefault('TT_LOGGER_LEVEL', 'error')       # hide tt-metal info spam

    print('capturing signposts under tracy\n')
    result = subprocess.run(cmd, cwd=cfg.PROJECT_ROOT, env=env)
    if result.returncode != 0:
        warnings.warn(f'python -m tracy exited {result.returncode}, so tracy_stats has no '
                      'results (e2e_latency is unaffected)', RuntimeWarning, stacklevel=2)
        return None

    reports = sorted(capture.glob('**/ops_perf_results*.csv'), key=lambda p: p.stat().st_mtime)
    return reports[-1] if reports else None


## ops_perf_results CSV -> (views, stage, iter, view, device_ms) rows.
## A stage's time is the sum of the kernel durations of every device op between its
## signposts, so nested ranges (subiter, iter, subtest) are sums of their stages.
def parse_ops_report(csv_path):
    open_ranges = {}        # (views, stage, iter, view) -> kernel ns so far
    timings = []

    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):

            ## Signpost rows open and close ranges
            if row.get(OP_TYPE) == 'signpost':
                match = SIGNPOST_PATTERN.match(row.get(OP_CODE, ''))
                if not match:
                    continue
                key = (int(match['views']), match['stage'], int(match['iter']), int(match['view']))
                if match['edge'] == 'start':
                    open_ranges[key] = 0.0
                elif key in open_ranges:
                    views, stage, index, view = key
                    timings.append((views, stage, index, view, open_ranges.pop(key) / 1e6))
                continue

            ## Device op: add its kernel time to every range that is open
            ns = row.get(KERNEL_NS)
            if not ns:
                continue        # host-only op, no device time
            for key in open_ranges:
                open_ranges[key] += float(ns)

    return timings
