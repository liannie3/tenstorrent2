# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            tt_parity.py
# CREATED:         11/23/2026
# REVISION:        1.0
# ======================================= #

"""Tenstorrent parity: every benchmark image vs the CPU fp32 golden file.

    python -m clipbench.tt.tt_parity

Same loop as common/golden.py, with the TT model (tt.yaml xla / trace settings,
so it checks exactly what the benchmark times). Writes parity.csv to the tt
results dir. Needs the golden file from common/golden.py.

The checks below are the same as in nvidia/nvidia_parity.py -- keep the two identical.
"""

import csv
from pathlib import Path

import numpy as np
import torch

from ..common import config as cfg
from ..common import data
from . import model_tt as model_lib

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


## Device tensor -> (tokens, dim) fp32 torch tensor on the CPU
def to_host(tensor):
    if not model_lib.XLA:
        tensor = model_lib.ttnn.to_torch(tensor)
    tensor = tensor.cpu().float()
    return tensor.reshape(-1, tensor.shape[-1])     # drops the batch dim(s): [1, 577, 1024] -> [577, 1024]


def main():
    config = cfg.load_config('tt')
    run = config['run']
    threshold = config['parity']['threshold'][run['precision']]

    golden = np.load(config['parity']['golden_path'])
    views_paths = data.load_views(config, len(config['data']['view_order']))   # all 6 cameras

    ## Card must be open before the model loads -- weights go straight to it
    model_lib.open_device(config)
    try:
        dtype = model_lib.resolve_dtype(run['precision'])          # depends on xla, so after open_device
        processor = model_lib.load_processor(config)
        encoder = model_lib.load_encoder(config)
        projector = model_lib.load_projector(config, encoder.vision_width)

        rows = []
        with torch.inference_mode():
            for camera, paths in views_paths.items():
                print(f'{camera:<16} {len(paths)} images', flush=True)
                for image_path in paths:
                    pixel_values = data.preprocess(processor, data.load_images([image_path]), 'cpu', torch.float32)
                    pixel_values = model_lib.to_device(pixel_values, dtype)
                    features = model_lib.select_image_features(
                        encoder(pixel_values, output_hidden_states=True), config)
                    embeddings = projector(features)

                    ## Finish the whole image on the card (one XLA graph), then copy both outputs back
                    model_lib.synchronize()
                    outputs = {'encoder': to_host(features), 'projector': to_host(embeddings)}
                    rows.append(check_image(f'{camera}/{image_path.stem}', outputs, golden, threshold))
    finally:
        model_lib.close_device()

    print(f"\nprofile: tt  precision: {run['precision']}  xla: {run.get('xla', False)}  trace: {run.get('trace', False)}")
    summarize(rows, threshold, Path(config['output']['results_dir']) / 'parity.csv')


if __name__ == '__main__':
    main()
