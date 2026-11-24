# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            golden.py
# CREATED:         11/23/2026
# REVISION:        1.0
# ======================================= #

"""Writes the fp32 CPU golden file that NVIDIA and TT outputs are compared against.

    python -m clipbench.common.golden

Output: parity.golden_path in base.yaml, one .npz with
    <camera>/<file stem>/encoder     (577, 1024) fp32   CLIP hidden layer fed to the projector
    <camera>/<file stem>/projector   (577, 2048) fp32   MLP projector output

"""

import zipfile
from pathlib import Path

import numpy as np
import torch

from . import config as cfg
from . import data
from ..nvidia import model_nvidia as model_lib


def main():
    config = cfg.load_config('cpu')

    processor = model_lib.load_processor(config)
    encoder = model_lib.load_encoder(config)
    projector = model_lib.load_projector(config, encoder.config.hidden_size)

    views_paths = data.load_views(config, len(config['data']['view_order']))   # all 6 cameras

    path = Path(config['parity']['golden_path'])
    path.parent.mkdir(parents=True, exist_ok=True)

    ## Written into the .npz one array at a time so ~5 GB never sits in RAM
    with zipfile.ZipFile(path, 'w', allowZip64=True) as zf, torch.inference_mode():
        for camera, paths in views_paths.items():
            print(f'{camera:<16} {len(paths)} images')
            for image_path in paths:
                pixel_values = data.preprocess(processor, data.load_images([image_path]), 'cpu', torch.float32)
                features = model_lib.select_image_features(
                    encoder(pixel_values, output_hidden_states=True), config)
                embeddings = projector(features)

                key = f'{camera}/{image_path.stem}'
                for name, out in (('encoder', features), ('projector', embeddings)):
                    with zf.open(f'{key}/{name}.npy', 'w', force_zip64=True) as f:
                        np.lib.format.write_array(f, out[0].numpy())

    print(f'\ngolden: {path}')


if __name__ == '__main__':
    main()
