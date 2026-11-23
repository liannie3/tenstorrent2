# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            schema.py
# CREATED:         11/13/2026
# REVISION:        1.0
# ======================================= #

"""Files written by tt/run.py: one CSV per stage, plus a manifest.

Same columns as nvidia/schema.py, so the analysis notebooks read both the same way.
"""

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import transformers

## ttnn may not be installed in the TT-XLA venv
try:
    import ttnn
except ImportError:
    ttnn = None

## view is the camera index within an iteration, or -1 for a row that spans the
## whole rig ('iter' and 'subtest')
COLUMNS = ('views', 'iter', 'view', 'duration_ms')


## One CSV per stage, named after the stage
def write_stage_csvs(timings, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    by_stage = {}
    for views, stage, index, view, duration_ms in timings:
        by_stage.setdefault(stage, []).append((views, index, view, duration_ms))

    paths = {}
    for stage, rows in by_stage.items():
        rows.sort()                 # view count, then iteration, then view
        path = directory / f'{stage}.csv'
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        paths[stage] = path

    return paths


## Short commit hash, or 'unknown' outside a git checkout
def _git_sha():
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return 'unknown'


## Written beside the CSVs so a result traces back to the code that made it
def write_manifest(config, images_by_views, path):
    manifest = {
        'profile': config['profile'],
        'platform': config['run']['target'],
        'device_id': config['run']['device_id'],
        'precision': config['run']['precision'],
        'xla': config['run'].get('xla', False),
        'model': config['model']['name'],
        'image_processor': config['model'].get('image_processor'),
        'layer_index': config['model']['layer_index'],
        'keep_cls': config['model']['keep_cls'],
        'views': config['run']['views'],
        'view_order': config['data']['view_order'],
        'images_by_views': images_by_views,
        'git_sha': _git_sha(),
        'run_time': datetime.now().isoformat(timespec='seconds'),
        'ttnn': getattr(ttnn, '__version__', 'unknown'),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path
