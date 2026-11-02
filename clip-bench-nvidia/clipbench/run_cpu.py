# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            run_cpu.py
# CREATED:         10/19/2026
# REVISION:        1.0
# ======================================= #

"""CPU runner. Times every stage in-process with time.perf_counter_ns.

    python -m clipbench.run_cpu --profile cpu

run_nvidia.py is a separate file on purpose: it measures a different way (NVTX
ranges read back by nsys), and the timing calls are the only thing that differs,
so they sit right next to the work they measure instead of behind a wrapper.
"""

import argparse
import time
from pathlib import Path

import torch

from . import config as cfg
from . import data
from . import model as model_lib
from .reports.schema_cpu import write_manifest, write_stage_csvs


## One sub-test: every frame, at a single view count. One iteration is frame i
## from each view, run one image at a time.
def run_subtest(config, encoder, processor, projector, views_paths, views, timings):

    ## Loads parameters which are initialized in the cpu.yaml file
    device = config['run']['device']                                             # config['run']['device'] returns CPU
    dtype = model_lib.resolve_dtype(config['run']['precision'])                  # Goes to model.py and makes sure config dtype is allowed
    max_frames = config['run'].get('max_frames')                                 # Caps the iterations
    images_seen = 0                                                              # Image counter

    num_frames = data.num_frames(views_paths)                                    # Iterations available, set by the shortest view
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    ## inference_mode wraps the whole sub-test because nothing here ever trains
    with torch.inference_mode():

        subtest_start = time.perf_counter_ns()   # Subtest initial time t0

        ## Loop over every frame, then over every view of that frame
        for i, frame_paths in enumerate(data.iter_frames(views_paths)):
            if i >= num_frames:
                break

            iter_start = time.perf_counter_ns()                                  # Time Marker: Start Iteration

            for view, path in enumerate(frame_paths):

                ## Decode
                t0 = time.perf_counter_ns()                                      # Time Marker: Start Load Images
                images = data.load_images([path])                                # Return: Returns a list of the path and pixel data of images

                ## Preprocess
                t1 = time.perf_counter_ns()                                      # Time Marker: End Load Images, Start Preprocessing                             (900, 1600, 3) uint8
                pixel_values = data.preprocess(processor, images, device, dtype) # Return: Resize: Converts 1600x900px res image to 597x336px                    (336, 597, 3) uint8
                                                                                 #         Center: Crops to 336x336px                                            (336, 336, 3) uint8
                                                                                 #         Rescale: Converts uint8 integers to floats [0, 255] --> [0.0, 1.0]    (336, 336, 3) float32
                                                                                 #         Normalize: Each R,G,B gets a different mean and STD                   (336, 336, 3) float32
                                                                                 #         Stack: Stacks the images in the NCHW tensor                           (1, 3, 336, 336) float32
                                                                                 #         To: Moves to device and casts precision (fp32 cpu / fp16 jetson)      (1, 3, 336, 336) float32 / float16
                ## Encode
                t2 = time.perf_counter_ns()                                      # Time Marker: End of preprocessing, Start Encode
                outputs = encoder(pixel_values, output_hidden_states=True)       # Return:
                features = model_lib.select_image_features(outputs, config)      # Return:

                ## Project
                t3 = time.perf_counter_ns()                                      # Time Marker: End of encode, Start Projection
                embeddings = projector(features)                                 # Return:
                t4 = time.perf_counter_ns()                                      # Time Marker: End of Projection and Sub-iteration

                ## Keeps track of all of the Stage timings with the iteration and
                ## view present so we can keep track of the image number
                timings.append((views, 'decode',     i, view, (t1 - t0) / 1e6))
                timings.append((views, 'preprocess', i, view, (t2 - t1) / 1e6))
                timings.append((views, 'encode',     i, view, (t3 - t2) / 1e6))
                timings.append((views, 'project',    i, view, (t4 - t3) / 1e6))
                timings.append((views, 'subiter',    i, view, (t4 - t0) / 1e6))

                images_seen += 1

            ## view -1 means all views: the whole rig for frame i. Everything is
            ## synchronous on the CPU, so this is just the sub-iterations summed.
            timings.append((views, 'iter', i, -1,
                            (time.perf_counter_ns() - iter_start) / 1e6))

        subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6

    ## One envelope row per view count -- this is what throughput comes from
    timings.append((views, 'subtest', 0, -1, subtest_ms))

    ## Loop variables outlive the loop and every sub-iteration has the same
    ## shapes, so the last one describes them all. Printed out here rather than
    ## inside the loop so console I/O stays outside the timed region.
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen


def main(argv=None):

    ## Reads from command line (important to have --profile when running program)
    parser = argparse.ArgumentParser(description='CLIP encoder + MLP projector sweep, CPU timing')
    parser.add_argument('--profile', default='cpu', choices=cfg.available_profiles(),
                        help='config profile from configs/ (default: cpu)')
    args = parser.parse_args(argv)

    ## 1. Config
    config = cfg.load_config(args.profile)
    run = config['run']

    ## A wall clock cannot read a CUDA stage without draining the queue first,
    ## which is the whole reason the NVIDIA runner exists. Refuse rather than
    ## publish numbers that only measure how long a synchronize took.
    if run['target'] != 'cpu':
        raise SystemExit(
            f"profile {args.profile!r} has run.target {run['target']!r} - run it with:\n"
            f'  python -m clipbench.run_nvidia --profile {args.profile}'
        )

    print(f"profile: {config['profile']}  device: {run['device']}  "
          f"precision: {run['precision']}")

    ## 2. Model pieces, built once and reused by every sub-test
    processor = model_lib.load_processor(config)                                # from model.py
    encoder = model_lib.load_encoder(config)                                    # from model.py
    projector = model_lib.load_projector(config, encoder.config.hidden_size)    # from model.py

    ## 3. One image list per camera. Loaded at the full view count just to fail
    ## early on a bad data.path -- each sub-test loads its own at step 5.
    all_views = data.load_views(config, max(run['views']))
    for name, view_paths in all_views.items():
        print(f'  {name:<16} {len(view_paths)} images')
    if not data.num_frames(all_views):
        raise SystemExit('no images found - check data.path and data.view_order in '
                         f"configs/base.yaml ({config['data']['path']})")

    ## 4. The timing buffer. Appended to all through the sweep below, read once
    ## at step 5 and not before.
    timings = []        # (views, stage, iter, view, duration_ms)

    ## 5. One sub-test per view count
    images_by_views = {}
    for views in run['views']:
        views_paths = data.load_views(config, views)
        print(f'\n=== sub-test: {data.num_frames(views_paths)} frames x {views} views ===')
        images_seen = run_subtest(
            config, encoder, processor, projector, views_paths, views, timings)
        images_by_views[views] = images_seen
        print(f'  images      : {images_seen}')

    ## 6. Sweep is over, so now the buffer gets read and the files get written.
    ## The manifest records how the run was configured so a CSV can be traced
    ## back to the code and versions that produced it.
    results_dir = Path(config['output']['results_dir'])
    print('\nmanifest:', write_manifest(config, images_by_views,
                                        results_dir / 'manifest.json'))
    for stage, path in write_stage_csvs(timings, results_dir).items():
        print(f'{stage:>10}:', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
