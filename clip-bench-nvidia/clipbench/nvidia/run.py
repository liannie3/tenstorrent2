# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            run.py
# CREATED:         10/19/2026
# REVISION:        2.0
# ======================================= #

"""NVIDIA runner.

    sudo nvpmodel -m 0        # MAXN -- do this before benchmarking
    sudo jetson_clocks        # pin clocks so sub-tests are comparable

    python -m clipbench.nvidia.run --profile jetson

Runs four tests, each into its own folder of
analysis/results/nvidia/benchmark_<M-D-YY>/eager/ (cudagraph/ with cuda_graph: true):

  cpu_latency    perf_counter on all four stages. Host time throughout, so
                 encode and project read launch cost, not GPU time.
  gpu_latency    CUDA events on encode and project. GPU execution time.
                 decode and preprocess are absent -- they launch no kernels.
  hardware_perf  tegrastats over the same sweep. No stage timings; the loop
                 is there to load the board while power, clocks and temps
                 are sampled.
  deep_metrics   the same pipeline in NVTX ranges, collected by relaunching
                 this module under `nsys profile`. The capture stays in
                 deep_metrics/; its ranges are merged into cpu_latency/ as an
                 nvtx_ms column, since an NVTX range is a host clock. CUPTI adds
                 overhead to every kernel launch, which is why it cannot share
                 a process with the others.

There is nothing to choose: one invocation runs all four. The child nsys
launches is the one with NSYS_ENV_GUARD set, and that child runs deep_metrics.
Running twice on the same day overwrites that day's folder.
"""

import argparse
import warnings

import torch

from ..common import config as cfg
from ..common import data
from ..common import validate
from . import model_nvidia as model_lib
from .benchmark_tests import (cpu_pure_latency, deep_metrics, gpu_pure_latency,
                              hardware_perf)
from .schema import under_nsys, write_manifest, write_stage_csvs



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
                        help='config profile from clipbench/*/configs/ (default: jetson)')
    args = parser.parse_args(argv)

    ## 1. Config
    config = cfg.load_config(args.profile)
    run = config['run']

    ## One folder per day, then eager/ or cudagraph/ so the two never overwrite
    ## each other. Set here so deep_metrics (and the nsys child) see the same path.
    run_dir = cfg.use_dated_results_dir(
        config, 'cudagraph' if run.get('cuda_graph', False) else 'eager')

    ## Make sure the run is on an nvidia device
    if run['target'] != 'nvidia':
        raise SystemExit(
            f"profile {args.profile!r} has run.target {run['target']!r} - "
            'this runner only takes nvidia profiles'
        )

    ## Checks if CUDA is available
    if not torch.cuda.is_available():
        raise SystemExit(
            'clipbench.nvidia.run times the GPU stages with CUDA events and needs a working '
            'CUDA device, but torch.cuda.is_available() is False.\n'
            'The CPU is only used for the golden file:\n'
            '  python -m clipbench.common.golden'
        )

    ## 2. Every test, hardcoded. Which ones this process runs is decided in the
    ## loop at step 5, not here.
    ## Keys are the results folder names.
    tests = {'cpu_latency': cpu_pure_latency.run,
             'gpu_latency': gpu_pure_latency.run,
             'deep_metrics': deep_metrics.run,
             'hardware_perf': hardware_perf.run}

    ## Missing NVTX is a flag, not a failure -- every other test still runs
    try:
        deep_metrics.require_nvtx()
        nvtx_ok = True
    except RuntimeError as exc:
        nvtx_ok = False
        warnings.warn(f'NVTX unavailable, so deep_metrics is skipped: {exc}',
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
                         f"clipbench/common/configs/base.yaml ({config['data']['path']})")

    ## Input check (FSR 3.2.5.1) before anything is timed. The nsys child skips
    ## it -- the parent already checked the same images.
    if not under_nsys():
        validate.require_inputs(all_views, processor)

    ## 5. One test at a time, one sub-test per view count. Each test gets its
    ## own buffer and its own results subdirectory, so the stage names stay the
    ## same everywhere without one test overwriting another's CSVs.
    ##
    ## Warmup runs here rather than inside the tests, so it is outside every
    ## test's timed region by construction.
    manifests_ok = True

    for test_name, run_subtest in tests.items():
        ## deep_metrics records nothing unless nsys is collecting, and the other
        ## three must not pay CUPTI overhead. So the child nsys launches runs
        ## that one and only that one, and the parent runs the rest.
        if (test_name == 'deep_metrics') != under_nsys():
            continue
        if test_name == 'deep_metrics' and not nvtx_ok:
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

        out_dir = run_dir / test_name
        manifest_path = write_manifest(config, images_by_views, out_dir / 'manifest.json')
        print('\nmanifest:', manifest_path)
        manifests_ok &= validate.report_manifest(manifest_path, images_by_views)
        for stage, path in write_stage_csvs(timings, out_dir).items():
            print(f'{stage:>10}:', path)

    if under_nsys():
        return 0

    ## 7. Profiled pass. Relaunches this module under nsys; that child runs
    ## deep_metrics alone at step 5, and its ranges become the nvtx_ms column.
    ##
    ## The column goes into cpu_latency/, not gpu_latency/: an NVTX range is a
    ## host clock, so nvtx_ms is the same quantity perf_counter measures and
    ## belongs beside it.
    if nvtx_ok:
        rep_path = deep_metrics.capture_under_nsys(config, ['--profile', args.profile])
        if rep_path is not None:
            deep_metrics.merge_nvtx(rep_path, run_dir / 'cpu_latency')

    ## 8. Every test's files are in their folder (FSR 3.2.8). deep_metrics is
    ## expected even when NVTX or nsys was missing -- a run without it is incomplete.
    results_ok = validate.report_results(run_dir, expected_results(run['views']))

    return 0 if manifests_ok and results_ok else 1


## What each test must leave in its folder of the run folder
def expected_results(views_list):
    return {
        'cpu_latency': ['manifest.json', *(f'{stage}.csv' for stage in
                        ('decode', 'preprocess', 'encode', 'project', 'subiter', 'iter', 'subtest'))],
        'gpu_latency': ['manifest.json', *(f'{stage}.csv' for stage in
                        ('encode', 'project', 'subiter', 'iter', 'subtest'))],
        'hardware_perf': ['manifest.json', *(f'tegrastats_v{views}.log' for views in views_list)],
        'deep_metrics': ['nsys_capture.nsys-rep'],
    }


## Footnote on preprocess: data.preprocess uploads from pinned memory with
## non_blocking=True, so views no longer sync on the H2D copy. The host-side
## preprocess number is pure CPU work now. The copy itself runs on the stream
## between views and isn't inside any event pair -- it shows up in 'iter' only.
if __name__ == '__main__':
    raise SystemExit(main())
