# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            run_tt.py
# CREATED:         11/8/2026
# REVISION:        1.0
# ======================================= #

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
def warmup_loop(config, encoder, processor, projector, views_paths, iterations):
    if iterations <= 0:
        return

    device = config['run']['device']
    dtype = model_lib.resolve_dtype(config['run']['precision'])
    warm_frame = next(data.iter_frames(views_paths), ())   # frame 0 of every view

    ## warm_ names so nothing here shadows what the test's shape print reads
    with torch.inference_mode():
        for _ in range(iterations):
            for warm_path in warm_frame:
                warm_images = data.load_images([warm_path])
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
    processor = model_lib.load_processor(config)                                # from model.py
    encoder = model_lib.load_encoder(config)                                    # from model.py
    projector = model_lib.load_projector(config, encoder.config.hidden_size)    # from model.py

    ## 4. One image list per camera. Loaded at the full view count just to fail
    ## early on a bad data.path -- each sub-test loads its own at step 5.
    all_views = data.load_views(config, max(run['views']))
    for name, view_paths in all_views.items():
        print(f'  {name:<16} {len(view_paths)} images')
    if not data.num_frames(all_views):
        raise SystemExit('no images found - check data.path and data.view_order in '
                         f"configs/base.yaml ({config['data']['path']})")

    ## 5. One test at a time, one sub-test per view count. Each test gets its
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
        timings = []        # (views, stage, iter, view, duration_ms)
        images_by_views = {}

        for views in run['views']:
            views_paths = data.load_views(config, views)
            print(f'\n=== sub-test: {data.num_frames(views_paths)} frames x '
                  f'{views} views ===')
            warmup_loop(config, encoder, processor, projector, views_paths,
                        run.get('warmup_iters', 0))
            images_seen = run_subtest(
                config, encoder, processor, projector, views_paths, views, timings)
            images_by_views[views] = images_seen
            print(f'  images      : {images_seen}')

        ## 6. The child's output is the capture itself, so only the parent
        ## writes files -- and its CSVs must exist before step 7 can add a
        ## column to them. The manifest records how the run was configured,
        ## including the Jetson power mode, since DVFS makes two runs in
        ## different modes incomparable.
        if under_nsys():
            continue

        out_dir = results_dir / test_name
        print('\nmanifest:', write_manifest(config, images_by_views,
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