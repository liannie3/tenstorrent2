# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            schema_nvidia.py
# CREATED:         10/20/2026
# REVISION:        1.0
# ======================================= #

"""Files written by run_nvidia.py: one CSV per stage, plus a manifest.

Same columns as schema_cpu.py, plus an nvtx_ms column merged in from an nsys
capture and the Jetson power mode in the manifest.
"""

import csv
import json
import os
import subprocess
from pathlib import Path

import torch
import transformers

COLUMNS = ('batch_size', 'iter', 'duration_ms')
NVTX_COLUMN = 'nvtx_ms'

## run_nvidia.py sets this on the child it launches under `nsys profile`
NSYS_ENV_GUARD = 'CLIPBENCH_UNDER_NSYS'


def under_nsys():
    return os.environ.get(NSYS_ENV_GUARD) == '1'


## One CSV per stage, named after the stage
def write_stage_csvs(timings, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    by_stage = {}
    for batch_size, stage, index, duration_ms in timings:
        by_stage.setdefault(stage, []).append((batch_size, index, duration_ms))

    paths = {}
    for stage, rows in by_stage.items():
        rows.sort()                 # batch size, then iteration
        path = directory / f'{stage}.csv'
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        paths[stage] = path

    return paths


## Adds an nvtx_ms column to the CSVs write_stage_csvs() already wrote, keyed on
## (batch_size, iter). Rewrites in place; safe to run twice.
##
## nvtx_ms is host-side, so for encode and project it is launch time, not GPU
## time. decode, preprocess and subtest are host-bound and should agree with
## duration_ms.
def merge_nvtx_column(nvtx_timings, directory, column=NVTX_COLUMN):
    directory = Path(directory)

    by_stage = {}
    for batch_size, stage, index, duration_ms in nvtx_timings:
        by_stage.setdefault(stage, {})[(batch_size, index)] = duration_ms

    merged = {}
    for stage, lookup in by_stage.items():
        path = directory / f'{stage}.csv'
        if not path.is_file():
            continue

        lines = path.read_text().splitlines()
        if not lines:
            continue

        reader = csv.reader(lines)
        header = next(reader)[:len(COLUMNS)]        # drop a previous nvtx_ms

        rows = []
        for row in reader:
            row = row[:len(COLUMNS)]
            try:
                key = (int(row[0]), int(row[1]))
            except (IndexError, ValueError):
                continue
            value = lookup.get(key)
            ## blank, not 0, when the capture has no range for that iteration
            rows.append(row + ['' if value is None else f'{value:.6f}'])

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(header) + [column])
            writer.writerows(rows)
        merged[stage] = path

    return merged


## Short commit hash, or 'unknown' outside a git checkout
def _git_sha():
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return 'unknown'


## Jetson power mode, e.g. '0 (MAXN)'. Clocks depend on it, so two runs in
## different modes are not comparable. 'unknown' off-Jetson.
def _nvpmodel_mode():
    try:
        result = subprocess.run(['nvpmodel', '-q'],
                                capture_output=True, text=True, check=True)
    except Exception:
        return 'unknown'

    ## 'NV Power Mode: MAXN' on one line, the number on the next
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if line.startswith('NV Power Mode'):
            name = line.split(':', 1)[1].strip()
            if i + 1 < len(lines):
                return f'{lines[i + 1]} ({name})'
            return name
    return 'unknown'


## Written beside the CSVs so a result traces back to the code that made it
def write_manifest(config, images_by_batch_size, path):
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
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
        ## CUPTI overhead on every launch, so not the numbers to quote
        'nsys_profiled': under_nsys(),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path
