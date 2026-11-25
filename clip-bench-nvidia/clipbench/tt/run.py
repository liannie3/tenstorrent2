# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            run.py
# CREATED:         11/8/2026
# REVISION:        1.0
# ======================================= #

"""Tenstorrent runner.

    python -m clipbench.tt.run --profile tt                          # e2e_latency
    python -m clipbench.tt.run --profile tt --test hardware_perf     # hardware sampling, on its own
    python -m clipbench.tt.run --profile tt --test e2e_latency       # any one test (repeatable)

Tests, each into its own subdirectory of
output.results_dir/benchmark_<M-D-YY> (analysis/results/tt/benchmark_11-17-26/...):

  e2e_latency   perf_counter per frame and per sub-test, one sync per frame.
                No profiler attached -- these are the wall-clock numbers to quote.
  hardware_perf hw_monitor over the same sweep. No stage timings; the loop is there
                to load the card while temperature, power and voltage are sampled.

  tracy_stage   host perf_counter stages + device signposts, one batch per frame,
                double-buffered. Needs run.xla false; launch under tracy:
                python -m tracy -r -p -o analysis/results/tt/tracy -m clipbench.tt.run --test tracy_stage

hardware_perf and tracy_stage only run when asked for with --test.
"""

import argparse
from datetime import date
from pathlib import Path

import torch

from ..common import config as cfg
from ..common import data
from . import model_tt as model_lib
from .benchmark_tests import e2e_latency, hardware_perf
from .schema import write_manifest, write_stage_csvs

TEST_NAMES = ('e2e_latency', 'hardware_perf', 'tracy_stage')
DEFAULT_TESTS = ['e2e_latency']     # hardware_perf runs on its own via --test


## Warmup loop for each subtest. The first pass also compiles every kernel.
## profiling: empty the device profiler buffers after every pass so they don't overflow
def warmup_loop(config, encoder, processor, projector, views_paths, iterations, profiling=False):
    if iterations <= 0:
        return
    if profiling:
        import ttnn

    dtype = model_lib.resolve_dtype(config['run']['precision'])
    warm_frame = next(data.iter_frames(views_paths), ())   # frame 0 of every view

    ## warm_ names so nothing here shadows what the test's shape print reads
    with torch.inference_mode():
        for _ in range(iterations):
            for warm_path in warm_frame:
                warm_images = data.load_images([warm_path])
                warm_pixel_values = data.preprocess(processor, warm_images, 'cpu', torch.float32)
                warm_pixel_values = model_lib.to_device(warm_pixel_values, dtype)
                warm_outputs = encoder(warm_pixel_values, output_hidden_states=True)
                warm_features = model_lib.select_image_features(warm_outputs, config)
                projector(warm_features)
                if profiling:
                    model_lib.synchronize()
                    ttnn.ReadDeviceProfiler(model_lib.tt_device.get_device())

    model_lib.synchronize()


def main(argv=None):
    ## Reads from command line (important to have --profile when running program)
    parser = argparse.ArgumentParser(description='CLIP encoder + MLP projector sweep, Tenstorrent')
    parser.add_argument('--profile', default='tt', choices=cfg.available_profiles(),
                        help='config profile from clipbench/*/configs/ (default: tt)')
    parser.add_argument('--test', action='append', choices=TEST_NAMES,
                        help='test to run, repeatable (default: e2e_latency)')
    args = parser.parse_args(argv)
    selected = args.test or DEFAULT_TESTS

    ## 1. Config
    config = cfg.load_config(args.profile)
    run = config['run']

    ## One dated folder per run, e.g. analysis/results/tt/benchmark_11-17-26
    today = date.today()
    config['output']['results_dir'] = str(
        Path(config['output']['results_dir']) / f'benchmark_{today.month}-{today.day}-{today.year % 100}')

    ## Make sure the run is on a Tenstorrent device
    if run['target'] != 'tenstorrent':
        raise SystemExit(f"profile {args.profile!r} has run.target {run['target']!r}, "
                         'expected tenstorrent')

    ## 2. Every test, hardcoded. --test picks which ones run.
    tests = {'e2e_latency': e2e_latency.run,
             'hardware_perf': hardware_perf.run}
    if 'tracy_stage' in selected:                         # needs ttnn + tracy, so imported only when asked for
        from .benchmark_tests import tracy_stage
        tests['tracy_stage'] = tracy_stage.run

    print(f"profile: {config['profile']}  device_id: {run['device_id']}  "
          f"precision: {run['precision']}  xla: {run.get('xla', False)}")

    ## 3. One image list per camera. Loaded at the full view count just to fail
    ## early on a bad data.path, before the card is opened.
    all_views = data.load_views(config, max(run['views']))
    for name, view_paths in all_views.items():
        print(f'  {name:<16} {len(view_paths)} images')
    if not data.num_frames(all_views):
        raise SystemExit('no images found - check data.path and data.view_order in '
                         f"clipbench/common/configs/base.yaml ({config['data']['path']})")

    results_dir = Path(config['output']['results_dir'])

    ## 4. Model pieces. The card must be open first -- weights go straight to it.
    print('\nopening device...', flush=True)
    model_lib.open_device(config)
    try:
        print('loading model + copying weights to the card...', flush=True)
        processor = model_lib.load_processor(config)
        encoder = model_lib.load_encoder(config)
        projector = model_lib.load_projector(config, encoder.vision_width)     # 1024 for ViT-L
        print('model loaded', flush=True)

        warmup_iters = run.get('warmup_iters', 0)

        ## 5. One test at a time, one sub-test per view count
        for test_name in selected:
            run_subtest = tests[test_name]

            print(f'\n########## {test_name} ##########')
            timings = []        # (views, stage, iter, view, duration_ms)
            images_by_views = {}

            for views in run['views']:
                views_paths = data.load_views(config, views)
                print(f'\n=== sub-test: {data.num_frames(views_paths)} frames x '
                      f'{views} views ===')
                print(f'  warmup      : {warmup_iters} iters...', flush=True)
                warmup_loop(config, encoder, processor, projector, views_paths, warmup_iters,
                            profiling=(test_name == 'tracy_stage'))
                images_seen = run_subtest(
                    config, encoder, processor, projector, views_paths, views, timings)
                images_by_views[views] = images_seen
                print(f'  images      : {images_seen}')

            out_dir = results_dir / test_name
            print('\nmanifest:', write_manifest(config, images_by_views, out_dir / 'manifest.json'))
            for stage, path in write_stage_csvs(timings, out_dir).items():
                print(f'{stage:>10}:', path)
    finally:
        model_lib.close_device()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
