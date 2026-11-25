# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            validate.py
# CREATED:         11/25/2026
# REVISION:        1.0
# ======================================= #

"""Validation checks shared by the NVIDIA and TT harnesses (FSR 3.2.5 - 3.2.8)."""

import json
from pathlib import Path

import torch
from PIL import Image

RAW_SIZE = (1600, 900)              # nuScenes camera frame, (width, height)
MODEL_INPUT = (1, 3, 336, 336)      # one preprocessed image, NCHW

## CLIP normalization of [0, 1] pixels: (0 - 0.4815) / 0.2686 = -1.79, (1 - 0.4082) / 0.2758 = 2.15
NORM_RANGE = (-1.80, 2.15)


## FSR 3.2.5.1 Input Integrity. Returns a list of failures, empty = pass.
def check_inputs(views_paths, processor):
    fails = []

    counts = {name: len(paths) for name, paths in views_paths.items()}
    if len(set(counts.values())) > 1:
        fails.append(f'camera views have different image counts: {counts}')

    for paths in views_paths.values():
        for path in paths:
            try:
                with Image.open(path) as img:
                    img.verify()                            # truncated / corrupt file
                with Image.open(path) as img:
                    size = img.size
                    img = img.convert('RGB')
            except Exception as exc:
                fails.append(f'{path}: cannot decode ({exc})')
                continue

            if size != RAW_SIZE:
                fails.append(f'{path}: {size[0]}x{size[1]}, expected {RAW_SIZE[0]}x{RAW_SIZE[1]}')
                continue

            x = processor(images=[img], return_tensors='pt')['pixel_values']
            if tuple(x.shape) != MODEL_INPUT or x.dtype != torch.float32:
                fails.append(f'{path}: preprocessed to {tuple(x.shape)} {x.dtype}, '
                             f'expected {MODEL_INPUT} torch.float32')
            elif not torch.isfinite(x).all():
                fails.append(f'{path}: preprocessed tensor has NaN/Inf')
            elif x.min() < NORM_RANGE[0] or x.max() > NORM_RANGE[1]:
                fails.append(f'{path}: values {x.min():.2f} to {x.max():.2f}, '
                             f'outside {NORM_RANGE} -- normalized wrong?')

    return fails


## Prints PASS, or every failure and stops the run before anything is timed
def require_inputs(views_paths, processor):
    total = sum(len(paths) for paths in views_paths.values())
    print(f'\ninput check : {total} images across {len(views_paths)} views...', flush=True)

    fails = check_inputs(views_paths, processor)
    if not fails:
        print(f'input check : PASS ({total}/{total} images)')
        return

    print(f'input check : FAIL ({len(fails)} problems)')
    for line in fails:
        print(f'  {line}')
    raise SystemExit('input check failed -- fix the images above before benchmarking')


## Fields both platforms' write_manifest() record
MANIFEST_FIELDS = ('profile', 'platform', 'precision', 'model', 'image_processor',
                   'layer_index', 'keep_cls', 'views', 'view_order', 'images_by_views',
                   'git_sha', 'torch', 'transformers')


## FSR 3.2.8.1 Run Manifest. Reads the file back from disk, so it checks what
## was actually written. images_by_views = what the run just did. Returns failures.
def check_manifest(path, images_by_views):
    path = Path(path)
    if not path.is_file():
        return [f'{path}: not written']
    try:
        manifest = json.loads(path.read_text())
    except ValueError as exc:
        return [f'{path}: not valid JSON ({exc})']

    fails = [f'missing field: {key}' for key in MANIFEST_FIELDS
             if manifest.get(key) in (None, '', [], {})]

    if manifest.get('git_sha') == 'unknown':
        fails.append('git_sha is unknown -- run from inside the git checkout')

    ## JSON turns the int view counts into strings
    expected = {str(views): seen for views, seen in images_by_views.items()}
    if manifest.get('images_by_views') != expected:
        fails.append(f"images_by_views {manifest.get('images_by_views')} != run's {expected}")

    return fails


## FSR 3.2.8 Run Traceability: every test's files landed in run_dir/<test folder>.
## expected = {folder: [file names or glob patterns]}. Returns failures.
def check_results(run_dir, expected):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return [f'{run_dir}: run folder missing']

    fails = []
    for folder, patterns in expected.items():
        out = run_dir / folder
        if not out.is_dir():
            fails.append(f'{out}: folder missing')
            continue
        for pattern in patterns:
            hits = list(out.glob(pattern))
            if not hits:
                fails.append(f'{out / pattern}: not written')
            fails += [f'{p}: empty file' for p in hits if p.stat().st_size == 0]

    ## Anything loose in the run folder was written to the wrong place
    stray = sorted(p.name for p in run_dir.iterdir() if p.is_file())
    if stray:
        fails.append(f'files outside any test folder in {run_dir}: {stray}')

    return fails


## Prints PASS or the failures. Returns True on pass.
def report_results(run_dir, expected):
    fails = check_results(run_dir, expected)
    if not fails:
        print(f'\nresults check: PASS ({len(expected)} test folders in {run_dir})')
        return True

    print(f'\nresults check: FAIL ({len(fails)} problems) -- this run is invalid')
    for line in fails:
        print(f'  {line}')
    return False


## Prints PASS or the failures. Returns True on pass; the run keeps going either way.
def report_manifest(path, images_by_views):
    fails = check_manifest(path, images_by_views)
    if not fails:
        print('manifest check: PASS')
        return True

    print(f'manifest check: FAIL ({len(fails)} problems) -- this run is invalid')
    for line in fails:
        print(f'  {line}')
    return False
