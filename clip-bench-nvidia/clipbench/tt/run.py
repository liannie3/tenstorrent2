# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            run.py
# CREATED:         11/8/2026
# REVISION:        1.0
# ======================================= #

"""Tenstorrent runner.

Tests, each into its own folder of analysis/results/tt/benchmark_<M-D-YY>/xla/
(ttnn/ with run.xla false). Running the same model again the same day overwrites it:

  e2e_latency   perf_counter per frame and per sub-test, one sync per frame.
                python -m clipbench.tt.run --profile tt

  hardware_perf hw_monitor over the same sweep. No stage timings; the loop is there
                to load the card while temperature, power and voltage are sampled.
                python -m clipbench.tt.run --profile tt --test hardware_perf

  tracy_stage   same loop and ranges as NVIDIA's nvtx_stats, as Tracy zones.
                No CSVs; read the .tracy file. Needs run.xla false; launch under tracy:
                TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000 python -m tracy -r -p --sync-host-device -o analysis/results/tt/benchmark_$(date +%-m-%-d-%y)/ttnn/tracy_stage -m clipbench.tt.run --test tracy_stage

hardware_perf and tracy_stage only run when asked for with --test.
"""

import argparse

import torch

from ..common import config as cfg
from ..common import data
from ..common import validate
from . import model_tt as model_lib
from .benchmark_tests import e2e_latency, hardware_perf
from .schema import write_manifest, write_stage_csvs

TEST_NAMES = ('e2e_latency', 'hardware_perf', 'tracy_stage')
DEFAULT_TESTS = ['e2e_latency']     # hardware_perf runs on its own via --test


TRACY_WARMUP_ITERS = 1      # one pass compiles every kernel; more only grows the .tracy file


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

    ## One folder per day, then xla/ or ttnn/ (the ported model) so the two
    ## never overwrite each other, e.g. analysis/results/tt/benchmark_11-17-26/xla
    run_dir = cfg.use_dated_results_dir(config, 'xla' if run.get('xla', False) else 'ttnn')

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

    ## Input validation check (FSR 3.2.5.1), before the card is opened. The processor is
    ## CPU-only, so it loads here rather than with the weights.
    processor = model_lib.load_processor(config)
    validate.require_inputs(all_views, processor)

    ## 4. Model pieces. The card must be open first -- weights go straight to it.
    print('\nopening device...', flush=True)
    model_lib.open_device(config)
    try:
        print('loading model + copying weights to the card...', flush=True)
        encoder = model_lib.load_encoder(config)
        projector = model_lib.load_projector(config, encoder.vision_width)     # 1024 for ViT-L
        print('model loaded', flush=True)

        ## Empty the device profiler buffers of the weight-load ops before the first forward pass
        if 'tracy_stage' in selected:
            import ttnn
            model_lib.synchronize()
            ttnn.ReadDeviceProfiler(model_lib.tt_device.get_device())

        warmup_iters = run.get('warmup_iters', 0)
        manifests_ok = True     # any failed manifest check -> exit code 1

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
                ## Under tracy every warmup pass is followed by a slow profiler read, so keep it short
                profiling = test_name == 'tracy_stage'
                iters = min(warmup_iters, TRACY_WARMUP_ITERS) if profiling else warmup_iters
                print(f'  warmup      : {iters} iters...', flush=True)
                warmup_loop(config, encoder, processor, projector, views_paths, iters,
                            profiling=profiling)
                images_seen = run_subtest(
                    config, encoder, processor, projector, views_paths, views, timings)
                images_by_views[views] = images_seen
                print(f'  images      : {images_seen}')

            out_dir = run_dir / test_name
            manifest_path = write_manifest(config, images_by_views, out_dir / 'manifest.json')
            print('\nmanifest:', manifest_path)
            manifests_ok &= validate.report_manifest(manifest_path, images_by_views)
            for stage, path in write_stage_csvs(timings, out_dir).items():      # none for tracy_stage
                print(f'{stage:>10}:', path)
    finally:
        model_lib.close_device()

    ## Every selected test's files are in their folder (FSR 3.2.8)
    expected = expected_results(run['views'])
    results_ok = validate.report_results(run_dir, {t: expected[t] for t in selected})

    return 0 if manifests_ok and results_ok else 1


## What each test must leave in its folder of the run folder. tracy_stage's
## .tracy file goes wherever `python -m tracy -o` points, so only its manifest is checked.
def expected_results(views_list):
    return {
        'e2e_latency': ['manifest.json', 'iter.csv', 'subtest.csv'],
        'hardware_perf': ['manifest.json', *(f'hw_v{views}.csv' for views in views_list)],
        'tracy_stage': ['manifest.json'],
    }


if __name__ == '__main__':
    raise SystemExit(main())
