# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            runner.py
# CREATED:         10/16/2026
# REVISION:        1.0
# ======================================= #

"""The sweep loop. Shared by every target, branches on none of them.

    python -m clipbench.runner --profile cpu
    python -m clipbench.runner --profile jetson     # under nsys; see targets/nvidia.py
"""

import argparse
import json
import subprocess
from pathlib import Path

import torch
import transformers

from . import config as cfg
from . import data
from . import model as model_lib
from .reports.schema import print_table, write_csv
from .targets import get_target


## One sub-test: the whole image set at a single batch size. Returns how many
## images actually ran, which write_manifest records per batch size.
def run_subtest(config, target, encoder, processor, projector, paths, batch_size):
    dtype = model_lib.resolve_dtype(config['run']['precision'])
    max_batches = config['run'].get('max_batches')
    images_seen = 0

    ## inference_mode wraps the whole sub-test because nothing here ever trains
    with target.subtest(batch_size), torch.inference_mode():
        for i, batch_paths in enumerate(data.iter_batches(paths, batch_size)):
            if max_batches is not None and i >= max_batches:
                break

            ## Every stage is timed by the target, never by this file:
            ## perf_counter on CPU, NVTX ranges on the Orin
            with target.iteration(batch_size, i):
                with target.stage('decode', batch_size):
                    images = data.load_images(batch_paths)

                with target.stage('preprocess', batch_size):
                    pixel_values = data.preprocess(processor, images, target, dtype)

                with target.stage('encode', batch_size):
                    outputs = encoder(pixel_values, output_hidden_states=True)
                    features = model_lib.select_image_features(outputs, config)

                with target.stage('project', batch_size):
                    embeddings = projector(features)

            images_seen += len(batch_paths)

    ## Loop variables outlive the loop, and every iteration has the same shapes,
    ## so one look at the last batch describes them all. Printed out here rather
    ## than inside the loop so console I/O stays outside the timed sub-test.
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen


## Short commit hash, or 'unknown' when git is missing or this is not a checkout
def _git_sha():
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return 'unknown'


## Written beside every run so a result can be traced back to the code and
## versions that produced it -- the question README section 4's changelog
## column ("Affects results?") exists to answer. A .nsys-rep carries none of
## this, so it is also what a report reader needs to label its rows.
def write_manifest(config, target, images_by_batch_size, path):
    manifest = {
        'profile': config['profile'],
        'platform': target.name,
        'device': config['run']['device'],
        'precision': config['run']['precision'],
        'model': config['model']['name'],
        'image_processor': config['model'].get('image_processor'),
        'layer_index': config['model']['layer_index'],
        'keep_cls': config['model']['keep_cls'],
        'batch_sizes': config['run']['batch_sizes'],
        'images_by_batch_size': images_by_batch_size,
        'git_sha': _git_sha(),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main(argv=None):
    ## Reads from command line (important to have --profile when running program)
    parser = argparse.ArgumentParser(description='CLIP encoder + MLP projector benchmark sweep') 
    parser.add_argument('--profile', default='cpu', choices=cfg.available_profiles(),
                        help='config profile from configs/ (default: cpu)')
    args = parser.parse_args(argv)

    ## 1. Config, and the target that owns all the hardware-specific behavior
    config = cfg.load_config(args.profile)
    run = config['run']
    target = get_target(config)
    print(f"profile: {config['profile']}  target: {target.name}  "
          f"device: {run['device']}  precision: {run['precision']}")

    ## 2. Model pieces, built once and reused by every sub-test
    processor = model_lib.load_processor(config)                                        # Loads processor from model.py
    encoder = model_lib.load_encoder(config, target)                                    # Loads encoder from model.py
    projector = model_lib.load_projector(config, target, encoder.config.hidden_size)    # Loads projector form model.py

    ## 3. The image list, identical for every sub-test
    paths = data.load_paths(config)                                                     # Paths for input images (list)
    print(f"found {len(paths)} images in {config['data']['path']}")                    
    if not paths:
        raise SystemExit('no images found - check data.path in configs/base.yaml')

    ## 4. One sub-test per batch size
    images_by_batch_size = {}
    for batch_size in run['batch_sizes']:
        print(f'\n=== sub-test: batch size {batch_size} over {len(paths)} images ===')
        images_seen = run_subtest( 
            config, target, encoder, processor, projector, paths, batch_size)
        images_by_batch_size[batch_size] = images_seen
        print(f'  images      : {images_seen}')

    ## 5. Results. The manifest is written for every target.
    ## Outputs how the run was configured so they can be traced back
    results_dir = Path(config['output']['results_dir'])
    print('\nmanifest:', write_manifest(config, target, images_by_batch_size,
                                        results_dir / 'manifest.json'))

    ## Checks if the target is NVIDIA or not CPU (should only execute under --profile jetson)
    if not target.reports_in_process:
        print(f'\n{target.name} target emits NVTX ranges only - no in-process timings.')
        print('Record the run under Nsight Systems to get durations:\n')
        print(f'  nsys profile --trace=cuda,nvtx,osrt --output {results_dir / "sweep"} \\')
        print(f'       python -m clipbench.runner --profile {config["profile"]}')
        return 0

    rows = target.rows(images_by_batch_size)
    print('stages  :', write_csv(rows, results_dir / 'stages.csv'), '\n')
    print_table(rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
