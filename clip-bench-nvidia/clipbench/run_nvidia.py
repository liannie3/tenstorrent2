# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            run_nvidia.py
# CREATED:         10/19/2026
# REVISION:        2.0
# ======================================= #

"""NVIDIA runner.

    sudo nvpmodel -m 0        # MAXN -- do this before benchmarking
    sudo jetson_clocks        # pin clocks so sub-tests are comparable

    python -m clipbench.run_nvidia --profile jetson

Runs two tests, in two processes:

  pure_latency  perf_counter on the host stages, CUDA events on the GPU stages.
                Nothing else on the critical path, so these are the numbers to
                quote. Written as the stage CSVs.
  deep_metrics  the same pipeline in NVTX ranges, collected by relaunching this
                module under `nsys profile`, then merged into those CSVs as an
                nvtx_ms column. CUPTI adds overhead to every kernel launch,
                which is why it cannot share a process with the first test.

There is nothing to choose: one invocation runs both. The child nsys launches
is the one with NSYS_ENV_GUARD set, and that child is what runs the NVTX test.
"""

import argparse
import warnings
from pathlib import Path

import torch

from . import config as cfg
from . import data
from . import model as model_lib
from .benchmark_tests import deep_metrics, pure_latency
from .reports.schema_nvidia import under_nsys, write_manifest, write_stage_csvs



## Warmup loop for each subtest
def warmup_loop(config, encoder, processor, projector, paths, batch_size, iterations):
    if iterations <= 0:
        return

    device = config['run']['device']
    dtype = model_lib.resolve_dtype(config['run']['precision'])
    warm_paths = paths[:batch_size]                   # one batch is enough

    ## warm_ names so nothing here shadows what the test's shape print reads
    with torch.inference_mode():
        for _ in range(iterations):
            warm_images = data.load_images(warm_paths)
            warm_pixel_values = data.preprocess(processor, warm_images, device, dtype)
            warm_outputs = encoder(warm_pixel_values, output_hidden_states=True)
            warm_features = model_lib.select_image_features(warm_outputs, config)
            projector(warm_features)

    torch.cuda.synchronize()


def main(argv=None):
    ## Reads from command line (important to have --profile when running program)
    parser = argparse.ArgumentParser(description='CLIP encoder + MLP projector sweep, CUDA event timing')
    parser.add_argument('--profile', default='jetson', choices=cfg.available_profiles(),
                        help='config profile from configs/ (default: jetson)')
    args = parser.parse_args(argv)

    ## 1. Config
    config = cfg.load_config(args.profile)
    run = config['run']

    ## Make sure the run is on an nvidia device
    if run['target'] != 'nvidia':
        raise SystemExit(
            f"profile {args.profile!r} has run.target {run['target']!r} - run it with:\n"
            f'  python -m clipbench.run_cpu --profile {args.profile}'
        )

    ## Checks if CUDA is available
    if not torch.cuda.is_available():
        raise SystemExit(
            'run_nvidia.py times the GPU stages with CUDA events and needs a working '
            'CUDA device, but torch.cuda.is_available() is False.\n'
            'For a CPU-only run use:\n'
            '  python -m clipbench.run_cpu --profile cpu'
        )

    ## 2. Which test this process runs -- hardcoded. under_nsys() is true only
    ## in the child nsys launched at step 8, so CUPTI never touches pure_latency
    if under_nsys():
        test_name, run_subtest = 'nvtx', deep_metrics.run
    else:
        test_name, run_subtest = 'latency', pure_latency.run

    print(f"profile: {config['profile']}  device: {run['device']}  "
          f"precision: {run['precision']}  test: {test_name}")

    ## 3. Model pieces, built once and reused by every sub-test
    processor = model_lib.load_processor(config)                                 # from model.py
    encoder = model_lib.load_encoder(config)                                    # from model.py
    projector = model_lib.load_projector(config, encoder.config.hidden_size)    # from model.py

    ## 4. The image list, identical for every sub-test
    paths = data.load_paths(config)                                             # list of image paths
    print(f"found {len(paths)} images in {config['data']['path']}")
    if not paths:
        raise SystemExit('no images found - check data.path in configs/base.yaml')

    ## 5. The timing buffer. Appended to all through the sweep below, read once
    ## at step 7 and not before. The nvtx test leaves it empty -- its data is
    ## in the capture, not in here.
    timings = []        # (batch_size, stage, iter, duration_ms)

    ## 6. One sub-test per batch size. Warmup runs here rather than inside the
    ## test, so it is outside every test's timed region by construction.
    images_by_batch_size = {}
    for batch_size in run['batch_sizes']:
        print(f'\n=== sub-test: batch size {batch_size} over {len(paths)} images ===')
        warmup_loop(config, encoder, processor, projector, paths, batch_size,
                    run.get('warmup_iters', 0))
        images_seen = run_subtest(
            config, encoder, processor, projector, paths, batch_size, timings)
        images_by_batch_size[batch_size] = images_seen
        print(f'  images      : {images_seen}')

    ## 7. The child's output is the capture itself, so only the parent writes
    ## files -- and its CSVs must exist before step 8 can add a column to them.
    ## The manifest records how the run was configured, including the Jetson
    ## power mode, since DVFS makes two runs in different modes incomparable.
    if under_nsys():
        return 0

    results_dir = Path(config['output']['results_dir'])
    print('\nmanifest:', write_manifest(config, images_by_batch_size,
                                        results_dir / 'manifest.json'))
    for stage, path in write_stage_csvs(timings, results_dir).items():
        print(f'{stage:>10}:', path)

    ## 8. Profiled pass. Relaunches this module under nsys; that child takes the
    ## nvtx branch at step 2, and its ranges become the nvtx_ms column.
    try:
        deep_metrics.require_nvtx()
    except RuntimeError as exc:
        warnings.warn(f'skipping the nvtx test: {exc}', RuntimeWarning, stacklevel=2)
        return 0

    rep_path = deep_metrics.capture_under_nsys(config, ['--profile', args.profile])
    if rep_path is not None:
        deep_metrics.merge_nvtx(rep_path, results_dir)

    return 0


## Footnote on preprocess: data.preprocess ends in a .to(device) on a pageable
## tensor, and CUDA synchronizes the stream before a pageable H2D copy. So the
## preprocess number includes waiting for the previous iteration's encode to
## finish. Switching that copy to pinned memory with non_blocking=True would
## remove the wait and give a cleaner split -- but then preprocess becomes
## asynchronous too and needs its own event pair, like encode and project.
if __name__ == '__main__':
    raise SystemExit(main())
