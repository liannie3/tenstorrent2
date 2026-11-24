# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            tt_parity.py
# CREATED:         11/23/2026
# REVISION:        2.0
# ======================================= #

"""Tenstorrent parity: every benchmark image vs the CPU fp32 golden file.

    python -m clipbench.tt.tt_parity

Same loop as common/golden.py, with the TT model (tt.yaml xla / trace settings,
so it checks exactly what the benchmark times). Needs the golden file from
common/golden.py.

Prints PASS or FAIL and exits 0 / 1. PASS means:
  - every image in the golden file was run, and
  - the mean PCC vs golden is > MEAN_PCC_THRESHOLD for both the encoder and projector outputs.

Same checks as nvidia/nvidia_parity.py, but with a lower threshold: the TT stack is
newer and not as accurate yet.
"""

import numpy as np
import torch

from ..common import config as cfg
from ..common import data
from . import model_tt as model_lib

OUTPUTS = ('encoder', 'projector')
MEAN_PCC_THRESHOLD = 0.95       # nvidia_parity.py uses 0.999


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


## One image. outputs = {'encoder': (577, 1024) tensor, 'projector': (577, 2048) tensor}, on the CPU.
## Returns {output name: PCC}. A wrong shape or non-finite output gives NaN, which fails the mean.
def image_pcc(key, outputs, golden):
    result = {}
    for name in OUTPUTS:
        out = outputs[name].float()
        ref = torch.from_numpy(golden[f'{key}/{name}'])
        ok = tuple(out.shape) == tuple(ref.shape) and bool(torch.isfinite(out).all())
        result[name] = pcc(out, ref) if ok else float('nan')
    return result


def check(pccs, golden):
    expected = {k.rsplit('/', 1)[0] for k in golden.files}
    missing = expected - set(pccs)
    all_run = not missing

    print(f'\n=== parity vs golden (mean PCC > {MEAN_PCC_THRESHOLD}) ===')
    print(f'images run  : {len(pccs)}/{len(expected)}  -> {"PASS" if all_run else "FAIL"}')
    for key in sorted(missing)[:5]:
        print(f'  missing   : {key}')

    pcc_ok = True
    for name in OUTPUTS:
        mean = np.mean([p[name] for p in pccs.values()]) if pccs else float('nan')
        ok = bool(mean > MEAN_PCC_THRESHOLD)     # NaN compares False
        pcc_ok &= ok
        print(f'{name:<10}  : mean PCC {mean:.6f}  -> {"PASS" if ok else "FAIL"}')

    passed = all_run and pcc_ok
    print(f'\nresult      : {"PASS" if passed else "FAIL"}')
    return passed


## Device tensor -> (tokens, dim) fp32 torch tensor on the CPU
def to_host(tensor):
    if not model_lib.XLA:
        tensor = model_lib.ttnn.to_torch(tensor)
    tensor = tensor.cpu().float()
    return tensor.reshape(-1, tensor.shape[-1])     # drops the batch dim(s): [1, 577, 1024] -> [577, 1024]


def main():
    config = cfg.load_config('tt')
    run = config['run']

    golden = np.load(config['parity']['golden_path'])
    views_paths = data.load_views(config, len(config['data']['view_order']))   # all 6 cameras

    ## Card must be open before the model loads -- weights go straight to it
    model_lib.open_device(config)
    try:
        dtype = model_lib.resolve_dtype(run['precision'])          # depends on xla, so after open_device
        processor = model_lib.load_processor(config)
        encoder = model_lib.load_encoder(config)
        projector = model_lib.load_projector(config, encoder.vision_width)

        pccs = {}
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
                    key = f'{camera}/{image_path.stem}'
                    pccs[key] = image_pcc(key, outputs, golden)
    finally:
        model_lib.close_device()

    print(f"\nprofile: tt  precision: {run['precision']}  xla: {run.get('xla', False)}  trace: {run.get('trace', False)}")
    raise SystemExit(0 if check(pccs, golden) else 1)


if __name__ == '__main__':
    main()
