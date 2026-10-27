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

Runs four tests, each into its own subdirectory of output.results_dir:

  cpu_pure_latency  perf_counter on all four stages. Host time throughout, so
                    encode and project read launch cost, not GPU time.
  gpu_pure_latency  CUDA events on encode and project. GPU execution time.
                    decode and preprocess are absent -- they launch no kernels.
  hardware_perf     tegrastats over the same sweep. No stage timings; the loop
                    is there to load the board while power, clocks and temps
                    are sampled.
  nvtx_stats        the same pipeline in NVTX ranges, collected by relaunching
                    this module under `nsys profile`. Merged into
                    cpu_pure_latency/ as an nvtx_ms column, since an NVTX range
                    is a host clock. CUPTI adds overhead to every kernel launch,
                    which is why it cannot share a process with the others.

There is nothing to choose: one invocation runs all four. The child nsys
launches is the one with NSYS_ENV_GUARD set, and that child runs nvtx_stats.
"""

import argparse
import warnings
from pathlib import Path

import torch

from . import config as cfg
from . import data
from . import model as model_lib
from .benchmark_tests import (cpu_pure_latency, deep_metrics, gpu_pure_latency,
                              hardware_perf)
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

    ## 2. Every test, hardcoded. Which ones this process runs is decided in the
    ## loop at step 5, not here.
    tests = {'cpu_pure_latency': cpu_pure_latency.run,
             'gpu_pure_latency': gpu_pure_latency.run,
             'nvtx_stats': deep_metrics.run,
             'hardware_perf': hardware_perf.run}

    ## Missing NVTX is a flag, not a failure -- every other test still runs
    try:
        deep_metrics.require_nvtx()
        nvtx_ok = True
    except RuntimeError as exc:
        nvtx_ok = False
        warnings.warn(f'NVTX unavailable, so nvtx_stats is skipped: {exc}',
                      RuntimeWarning, stacklevel=2)

    print(f"profile: {config['profile']}  device: {run['device']}  "
          f"precision: {run['precision']}  nvtx: {'yes' if nvtx_ok else 'NO'}")

    ## 3. Model pieces, built once and reused by every sub-test
    processor = model_lib.load_processor(config)                                 # from model.py
    encoder = model_lib.load_encoder(config)                                    # from model.py
    projector = model_lib.load_projector(config, encoder.config.hidden_size)    # from model.py

    ## 4. The image list, identical for every sub-test
    paths = data.load_paths(config)                                             # list of image paths
    print(f"found {len(paths)} images in {config['data']['path']}")
    if not paths:
        raise SystemExit('no images found - check data.path in configs/base.yaml')

    ## 5. One test at a time, one sub-test per batch size. Each test gets its
    ## own buffer and its own results subdirectory, so the stage names stay the
    ## same everywhere without one test overwriting another's CSVs.
    ##
    ## Warmup runs here rather than inside the tests, so it is outside every
    ## test's timed region by construction.
    results_dir = Path(config['output']['results_dir'])

    for test_name, run_subtest in tests.items():
        ## nvtx_stats records nothing unless nsys is collecting, and the other
        ## three must not pay CUPTI overhead. So the child nsys launches runs
        ## that one and only that one, and the parent runs the rest.
        if (test_name == 'nvtx_stats') != under_nsys():
            continue
        if test_name == 'nvtx_stats' and not nvtx_ok:
            continue

        print(f'\n########## {test_name} ##########')
        timings = []        # (batch_size, stage, iter, duration_ms)
        images_by_batch_size = {}

        for batch_size in run['batch_sizes']:
            print(f'\n=== sub-test: batch size {batch_size} over {len(paths)} images ===')
            warmup_loop(config, encoder, processor, projector, paths, batch_size,
                        run.get('warmup_iters', 0))
            images_seen = run_subtest(
                config, encoder, processor, projector, paths, batch_size, timings)
            images_by_batch_size[batch_size] = images_seen
            print(f'  images      : {images_seen}')

        ## 6. The child's output is the capture itself, so only the parent
        ## writes files -- and its CSVs must exist before step 7 can add a
        ## column to them. The manifest records how the run was configured,
        ## including the Jetson power mode, since DVFS makes two runs in
        ## different modes incomparable.
        if under_nsys():
            continue

        out_dir = results_dir / test_name
        print('\nmanifest:', write_manifest(config, images_by_batch_size,
                                            out_dir / 'manifest.json'))
        for stage, path in write_stage_csvs(timings, out_dir).items():
            print(f'{stage:>10}:', path)

    if under_nsys():
        return 0

    ## 7. Profiled pass. Relaunches this module under nsys; that child runs
    ## nvtx_stats alone at step 5, and its ranges become the nvtx_ms column.
    ##
    ## The column goes into cpu_pure_latency/, not gpu_pure_latency/: an NVTX
    ## range is a host clock, so nvtx_ms is the same quantity perf_counter
    ## measures and belongs beside it.
    if not nvtx_ok:
        return 0

    rep_path = deep_metrics.capture_under_nsys(config, ['--profile', args.profile])
    if rep_path is not None:
        deep_metrics.merge_nvtx(rep_path, results_dir / 'cpu_pure_latency')

    return 0


## Footnote on preprocess: data.preprocess ends in a .to(device) on a pageable
## tensor, and CUDA synchronizes the stream before a pageable H2D copy. So the
## preprocess number includes waiting for the previous iteration's encode to
## finish. Switching that copy to pinned memory with non_blocking=True would
## remove the wait and give a cleaner split -- but then preprocess becomes
## asynchronous too and needs its own event pair, like encode and project.
if __name__ == '__main__':
    raise SystemExit(main())
