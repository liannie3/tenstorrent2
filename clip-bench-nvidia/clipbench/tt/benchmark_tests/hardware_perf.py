# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            hardware_perf.py
# CREATED:         11/17/2026
# REVISION:        1.0
# ======================================= #

## hw_monitor over the same sweep the other tests run. TT's version of nvidia/hardware_perf.py.
## No stage timings -- the loop exists only to give the card something to do while
## hw_monitor samples temperature, power, current and voltage.
##
## hw_monitor runs as its own process (like tegrastats), so sampling never competes
## with this loop for the GIL. Writes one CSV per view count and nothing else.

import signal
import subprocess
import sys
from pathlib import Path

import torch

from ...common import config as cfg
from ...common import data
from .. import model_tt as model_lib

## Same default as the NVIDIA test's tegrastats interval
INTERVAL_MS = 100


## One sub-test: every frame, at a single view count.
## timings is unused -- the samples are the data, and they are in the CSV.
## Kept so every test has the same signature.
def run(config, encoder, processor, projector, views_paths, views, timings):

    ## Loads parameters which are initialized in the tt.yaml file
    dtype = model_lib.resolve_dtype(config['run']['precision'])    # ttnn.bfloat16, or torch.bfloat16 under XLA
    max_frames = config['run'].get('max_frames')                   # Caps the iterations
    images_seen = 0                                                # Image counter

    num_frames = data.num_frames(views_paths)                      # Iterations available, set by the shortest view
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    ## One CSV per view count, beside the other tests' output
    out_dir = Path(config['output']['results_dir']) / 'hardware_perf'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'hw_v{views}.csv'
    interval = config['run'].get('hw_interval_ms', INTERVAL_MS)

    ## Start from an empty queue so warmup work isn't sampled as this sub-test's load
    model_lib.synchronize()

    sampler = subprocess.Popen([sys.executable, '-m', 'clipbench.tt.hw_monitor',
                                '--interval', str(interval), '--out', str(log_path), '--quiet'],
                               cwd=cfg.PROJECT_ROOT)
    print(f'  hw_monitor  : every {interval} ms -> {log_path}')

    try:
        with torch.inference_mode():
            for i, frame_paths in enumerate(data.iter_frames(views_paths)):
                if i >= num_frames:
                    break

                ## One sub-iteration per view, no synchronize between them
                for path in frame_paths:

                    ## Untimed on purpose -- hw_monitor is the only instrument here
                    images = data.load_images([path])
                    pixel_values = data.preprocess(processor, images, 'cpu', torch.float32)
                    pixel_values = model_lib.to_device(pixel_values, dtype)
                    outputs = encoder(pixel_values, output_hidden_states=True)
                    features = model_lib.select_image_features(outputs, config)
                    embeddings = projector(features)

                    images_seen += 1

                ## Frame sync: wait for all views' device work to finish
                model_lib.synchronize()

        ## Drains the queue so the last ops land inside the sampled window
        model_lib.synchronize()
    finally:
        ## in a finally so a SystemExit mid-sweep does not leave it running.
        ## SIGINT = Ctrl+C, so hw_monitor closes its CSV cleanly.
        sampler.send_signal(signal.SIGINT)
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.kill()

    ## Shapes are the same every iteration, so the last one describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen
