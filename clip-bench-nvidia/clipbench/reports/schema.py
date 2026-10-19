# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            schema.py
# CREATED:         10/16/2026
# REVISION:        3.0
# ======================================= #

"""Everything the harness writes to disk, written after the sweep has finished.

Both runners fill a timing buffer of
    (batch_size, stage, iter, duration_ms)
rows and hand the whole thing to write_stage_csvs() at the very end. run_cpu.py
measures every stage with perf_counter; run_nvidia.py measures the host stages
the same way and the GPU stages with CUDA events. Either way the rows come out
identical, so a jetson CSV and a cpu CSV have the same columns and can be
compared directly.

clipbench/reports/nsys.py builds the same rows out of an nsys capture, but
writes them into a nsys/ subdirectory so a profiled run never overwrites the
event timings it should be checked against.
"""

import csv
import json
import subprocess
from pathlib import Path

import torch
import transformers

## Columns of each per-stage CSV. The stage name is the file name, so it is not
## repeated in every row.
COLUMNS = ('batch_size', 'iter', 'duration_ms')


## One CSV per stage: decode.csv, preprocess.csv, encode.csv, project.csv,
## iter.csv, subtest.csv
def write_stage_csvs(timings, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    by_stage = {}
    for batch_size, stage, index, duration_ms in timings:
        by_stage.setdefault(stage, []).append((batch_size, index, duration_ms))

    paths = {}
    for stage, rows in by_stage.items():
        rows.sort()     ## batch size, then iteration
        path = directory / f'{stage}.csv'
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        paths[stage] = path

    return paths


## Short commit hash, or 'unknown' when git is missing or this is not a checkout
def _git_sha():
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return 'unknown'


## Jetson power mode, e.g. '0 (MAXN)'. The clocks the GPU and CPU are allowed
## to reach depend on this, so two runs in different modes are not comparable
## and the mode has to sit beside the numbers. 'unknown' on anything that is
## not a Jetson, since nvpmodel only exists there.
def _nvpmodel_mode():
    try:
        result = subprocess.run(['nvpmodel', '-q'],
                                capture_output=True, text=True, check=True)
    except Exception:
        return 'unknown'

    ## The output looks like
    ##     NV Power Mode: MAXN
    ##     0
    ## so the name is on the labelled line and the number on the one after it
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if line.startswith('NV Power Mode'):
            name = line.split(':', 1)[1].strip()
            if i + 1 < len(lines):
                return f'{lines[i + 1]} ({name})'
            return name
    return 'unknown'


## Written beside every run so a result can be traced back to the code and
## versions that produced it -- the question README section 4's changelog column
## ("Affects results?") exists to answer. A .nsys-rep carries none of this, so
## it is also what a report reader needs to label its rows.
def write_manifest(config, images_by_batch_size, path, subtest_windows=None):
    manifest = {
        'profile': config['profile'],
        'platform': config['run']['target'],
        'device': config['run']['device'],
        'precision': config['run']['precision'],
        'model': config['model']['name'],
        'image_processor': config['model'].get('image_processor'),
        'layer_index': config['model']['layer_index'],
        'keep_cls': config['model']['keep_cls'],
        'batch_sizes': config['run']['batch_sizes'],
        'images_by_batch_size': images_by_batch_size,
        'git_sha': _git_sha(),
        'nvpmodel_mode': _nvpmodel_mode(),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
    }

    ## Unix start/end time of each sub-test, written by the NVIDIA runner.
    ## tegrastats samples once every 100ms or slower, so it cannot be matched
    ## to a single stage, but a sub-test runs for seconds -- these windows are
    ## what a tegrastats log gets sliced on to get power per batch size.
    if subtest_windows is not None:
        manifest['subtest_windows'] = subtest_windows

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path
