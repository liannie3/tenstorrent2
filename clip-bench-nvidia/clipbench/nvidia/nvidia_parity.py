# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            nvidia_parity.py
# CREATED:         10/15/2026
# REVISION:        2.0
# ======================================= #

"""Jetson parity: every benchmark image vs the CPU fp32 golden file.

    python -m clipbench.nvidia.nvidia_parity

Same loop as common/golden.py, with the Jetson model (jetson.yaml precision and
cuda_graph setting, so it checks exactly what the benchmark times). Writes
parity.csv to the jetson results dir. Needs the golden file from common/golden.py.

The checks below are the same as in tt/tt_parity.py -- keep the two identical.
"""

import csv
from pathlib import Path

import numpy as np
import torch

from ..common import config as cfg
from ..common import data
from . import model_nvidia as model_lib

OUTPUTS = ('encoder', 'projector')


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


## One image. outputs = {'encoder': (577, 1024) tensor, 'projector': (577, 2048) tensor}, on the CPU.
## Returns one CSV row. Failures are recorded, not raised, so the run always covers every image.
def check_image(key, outputs, golden, threshold):
    row = {'key': key, 'camera': key.split('/')[0], 'pass': True}

    for name in OUTPUTS:
        out = outputs[name].float()
        ref = torch.from_numpy(golden[f'{key}/{name}'])

        shape_ok = tuple(out.shape) == tuple(ref.shape)
        finite = bool(torch.isfinite(out).all())
        row[f'{name}_shape'] = 'x'.join(map(str, out.shape))
        row[f'{name}_finite'] = finite

        if shape_ok and finite:
            row[f'{name}_pcc'] = pcc(out, ref)
            row[f'{name}_cosine'] = cosine(out, ref)
        else:
            row[f'{name}_pcc'] = row[f'{name}_cosine'] = float('nan')

        if not (shape_ok and finite and row[f'{name}_pcc'] >= threshold):
            row['pass'] = False

    return row


## Writes every row to a CSV and prints the numbers that go in the README
def summarize(rows, threshold, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(r['pass'] for r in rows)
    print(f'\n=== parity vs golden (PCC threshold {threshold}) ===')
    print(f'pass        : {passed}/{len(rows)}  -> {"PASS" if passed == len(rows) else "FAIL"}')
    for name in OUTPUTS:
        p = np.array([r[f'{name}_pcc'] for r in rows])
        c = np.array([r[f'{name}_cosine'] for r in rows])
        print(f'{name:<10}  : PCC min {np.nanmin(p):.6f}  mean {np.nanmean(p):.6f}   '
              f'cosine min {np.nanmin(c):.6f}  mean {np.nanmean(c):.6f}')

    ## Per camera, to spot a view that behaves differently
    print('\nper camera  : pass   min encoder PCC   min projector PCC')
    for camera in dict.fromkeys(r['camera'] for r in rows):
        cam = [r for r in rows if r['camera'] == camera]
        print(f'  {camera:<16} {sum(r["pass"] for r in cam):>3}/{len(cam):<3}  '
              f'{np.nanmin([r["encoder_pcc"] for r in cam]):.6f}          '
              f'{np.nanmin([r["projector_pcc"] for r in cam]):.6f}')

    ## Lowest PCC first; NaN (bad shape / NaN output) counts as worst
    def score(r):
        return np.nan_to_num(np.fmin(r['encoder_pcc'], r['projector_pcc']), nan=-np.inf)

    print('\nworst 5     :')
    for r in sorted(rows, key=score)[:5]:
        print(f'  {r["key"]}  encoder {r["encoder_pcc"]:.6f}  projector {r["projector_pcc"]:.6f}'
              f'  {"PASS" if r["pass"] else "FAIL"}')

    print(f'\ncsv: {csv_path}')
    return passed == len(rows)


def main():
    config = cfg.load_config('jetson')
    run = config['run']
    dtype = model_lib.resolve_dtype(run['precision'])
    threshold = config['parity']['threshold'][run['precision']]

    results_dir = Path(config['output']['results_dir'])
    if run.get('cuda_graph', False):
        results_dir = results_dir.with_name(results_dir.name + '_cudagraph')    # same as nvidia/run.py

    golden = np.load(config['parity']['golden_path'])

    processor = model_lib.load_processor(config)
    encoder = model_lib.load_encoder(config)
    projector = model_lib.load_projector(config, encoder.config.hidden_size)

    views_paths = data.load_views(config, len(config['data']['view_order']))   # all 6 cameras

    rows = []
    with torch.inference_mode():
        for camera, paths in views_paths.items():
            print(f'{camera:<16} {len(paths)} images', flush=True)
            for image_path in paths:
                pixel_values = data.preprocess(processor, data.load_images([image_path]), run['device'], dtype)
                features = model_lib.select_image_features(
                    encoder(pixel_values, output_hidden_states=True), config)
                embeddings = projector(features)

                ## Copied off the GPU right away -- a CUDA graph replay reuses its output buffers
                outputs = {'encoder': features[0].float().cpu(), 'projector': embeddings[0].float().cpu()}
                rows.append(check_image(f'{camera}/{image_path.stem}', outputs, golden, threshold))

    print(f"\nprofile: jetson  precision: {run['precision']}  cuda_graph: {run.get('cuda_graph', False)}")
    summarize(rows, threshold, results_dir / 'parity.csv')


if __name__ == '__main__':
    main()
