# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            run.py
# CREATED:         11/8/2026
# REVISION:        1.0
# ======================================= #

"""Tenstorrent runner.

    python -m clipbench.tt.run --profile tt                          # e2e_latency + tracy_stats
    python -m clipbench.tt.run --profile tt --test hardware_perf     # hardware sampling, on its own
    python -m clipbench.tt.run --profile tt --test e2e_latency       # any one test (repeatable)

Tests, each into its own subdirectory of
output.results_dir/benchmark_<M-D-YY> (analysis/results/tt/benchmark_11-17-26/...):

  e2e_latency   perf_counter per frame and per sub-test, one sync per frame.
                No profiler attached -- these are the wall-clock numbers to quote.
  hardware_perf hw_monitor over the same sweep. No stage timings; the loop is there
                to load the card while temperature, power and voltage are sampled.
  tracy_stats   signposts around every stage, collected by relaunching this module
                under `python -m tracy -r`. Stage time = device kernel time of the
                ops between a stage's signposts.

hardware_perf only runs when asked for with --test. The child tracy launches is the
one with TRACY_ENV_GUARD set, and that child runs tracy_stats only. The parent closes the
card before launching it, since only one process can hold a TT device.
"""

import argparse
import warnings
from datetime import date
from pathlib import Path

import torch
import ttnn

from ..common import config as cfg
from ..common import data
from . import model_tt as model_lib
from .benchmark_tests import e2e_latency, hardware_perf, tracy_stats
from .schema import write_manifest, write_stage_csvs

TEST_NAMES = ('e2e_latency', 'hardware_perf', 'tracy_stats')
DEFAULT_TESTS = ['e2e_latency', 'tracy_stats']     # hardware_perf runs on its own via --test


## Warmup loop for each subtest. The first pass also compiles every kernel.
def warmup_loop(config, encoder, processor, projector, views_paths, iterations):
    if iterations <= 0:
        return

    dtype = model_lib.resolve_dtype(config['run']['precision'])
    warm_frame = next(data.iter_frames(views_paths), ())   # frame 0 of every view

    ## warm_ names so nothing here shadows what the test's shape print reads
    with torch.inference_mode():
        for _ in range(iterations):
            for warm_path in warm_frame:
                warm_images = data.load_images([warm_path])
                warm_pixel_values = data.preprocess(processor, warm_images, 'cpu', torch.float32)
                warm_pixel_values = model_lib.to_device(warm_pixel_values, dtype)
                warm_outputs = encoder.forward(warm_pixel_values)
                warm_features = model_lib.select_image_features(warm_outputs, config)
                projector(warm_features)

    ttnn.synchronize_device(model_lib.tt_device.get_device())


def main(argv=None):
    ## Reads from command line (important to have --profile when running program)
    parser = argparse.ArgumentParser(description='CLIP encoder + MLP projector sweep, Tenstorrent')
    parser.add_argument('--profile', default='tt', choices=cfg.available_profiles(),
                        help='config profile from clipbench/*/configs/ (default: tt)')
    parser.add_argument('--results-dir', default=None,
                        help='exact output folder (set by the parent for the tracy child)')
    parser.add_argument('--test', action='append', choices=TEST_NAMES,
                        help='test to run, repeatable (default: e2e_latency + tracy_stats)')
    args = parser.parse_args(argv)
    selected = args.test or DEFAULT_TESTS

    ## 1. Config
    config = cfg.load_config(args.profile)
    run = config['run']

    ## One dated folder per run, e.g. analysis/results/tt/benchmark_11-17-26.
    ## The tracy child is handed the parent's folder so both write to the same place.
    if args.results_dir:
        config['output']['results_dir'] = args.results_dir
    else:
        today = date.today()
        config['output']['results_dir'] = str(
            Path(config['output']['results_dir']) / f'benchmark_{today.month}-{today.day}-{today.year % 100}')

    ## Make sure the run is on a Tenstorrent device
    if run['target'] != 'tenstorrent':
        raise SystemExit(f"profile {args.profile!r} has run.target {run['target']!r}, "
                         'expected tenstorrent')

    ## 2. Every test, hardcoded. --test picks which ones run.
    tests = {'e2e_latency': e2e_latency.run,
             'hardware_perf': hardware_perf.run,
             'tracy_stats': tracy_stats.run}

    ## Missing tracy is a flag, not a failure -- e2e_latency still runs
    try:
        tracy_stats.require_tracy()
        tracy_ok = True
    except RuntimeError as exc:
        tracy_ok = False
        warnings.warn(f'tracy unavailable, so tracy_stats is skipped: {exc}',
                      RuntimeWarning, stacklevel=2)

    under_tracy = tracy_stats.under_tracy()
    print(f"profile: {config['profile']}  device_id: {run['device_id']}  "
          f"precision: {run['precision']}  tracy: {'yes' if tracy_ok else 'NO'}")

    ## 3. One image list per camera. Loaded at the full view count just to fail
    ## early on a bad data.path, before the card is opened.
    all_views = data.load_views(config, max(run['views']))
    for name, view_paths in all_views.items():
        print(f'  {name:<16} {len(view_paths)} images')
    if not data.num_frames(all_views):
        raise SystemExit('no images found - check data.path and data.view_order in '
                         f"clipbench/common/configs/base.yaml ({config['data']['path']})")

    results_dir = Path(config['output']['results_dir'])

    ## The tracy child runs tracy_stats and only that; the parent runs the rest of --test
    if under_tracy:
        local_tests = ['tracy_stats']
    else:
        local_tests = [name for name in selected if name != 'tracy_stats']

    ## 4. Model pieces. The card must be open first -- weights go straight to it.
    ## Skipped when nothing runs in this process (--test tracy_stats alone).
    if local_tests:
        print('\nopening device...', flush=True)
        model_lib.open_device(config)
        try:
            print('loading model + copying weights to the card...', flush=True)
            processor = model_lib.load_processor(config)
            encoder = model_lib.load_encoder(config)
            projector = model_lib.load_projector(config, encoder.vision_width)     # 1024 for ViT-L
            print('model loaded', flush=True)

            ## Kernel times don't depend on warmup past the first compile, so the tracy pass keeps it short
            warmup_iters = run.get('tracy_warmup_iters', 2) if under_tracy else run.get('warmup_iters', 0)

            ## 5. One test at a time, one sub-test per view count
            for test_name in local_tests:
                run_subtest = tests[test_name]

                print(f'\n########## {test_name} ##########')
                timings = []        # (views, stage, iter, view, duration_ms)
                images_by_views = {}

                for views in run['views']:
                    views_paths = data.load_views(config, views)
                    print(f'\n=== sub-test: {data.num_frames(views_paths)} frames x '
                          f'{views} views ===')
                    print(f'  warmup      : {warmup_iters} iters...', flush=True)
                    warmup_loop(config, encoder, processor, projector, views_paths, warmup_iters)
                    images_seen = run_subtest(
                        config, encoder, processor, projector, views_paths, views, timings)
                    images_by_views[views] = images_seen
                    print(f'  images      : {images_seen}')

                ## The tracy child's output is the ops report, so it writes nothing here
                if under_tracy:
                    continue

                out_dir = results_dir / test_name
                print('\nmanifest:', write_manifest(config, images_by_views, out_dir / 'manifest.json'))
                for stage, path in write_stage_csvs(timings, out_dir).items():
                    print(f'{stage:>10}:', path)
        finally:
            model_lib.close_device()

    if under_tracy or not tracy_ok or 'tracy_stats' not in selected:
        return 0

    ## 6. Profiled pass. The card is closed now, so the child can open it.
    report = tracy_stats.capture_under_tracy(
        config, ['--profile', args.profile, '--results-dir', str(results_dir)])
    if report is None:
        warnings.warn('tracy wrote no ops_perf_results CSV', RuntimeWarning, stacklevel=2)
        return 0

    timings = tracy_stats.parse_ops_report(report)
    if not timings:
        warnings.warn(f'no clipbench signposts in {report}', RuntimeWarning, stacklevel=2)
        return 0

    ## images_by_views isn't known in the parent for this pass, so count it from the rows
    images_by_views = {}
    for views, stage, _, _, _ in timings:
        if stage == 'subiter':
            images_by_views[views] = images_by_views.get(views, 0) + 1

    out_dir = results_dir / 'tracy_stats'
    print('\nmanifest:', write_manifest(config, images_by_views, out_dir / 'manifest.json',
                                        tracy_profiled=True))
    for stage, path in write_stage_csvs(timings, out_dir).items():
        print(f'{stage:>10}:', path)
    print('\nops report:', report)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
