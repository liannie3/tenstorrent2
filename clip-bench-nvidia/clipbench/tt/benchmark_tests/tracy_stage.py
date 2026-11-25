# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            tracy_stage.py
# CREATED:         11/22/2026
# REVISION:        4.0
# ======================================= #

## Host vs device stage timing, run under the Tracy profiler. Ported ttnn model only (run.xla false).
##
##   python -m tracy -r -p -o analysis/results/tt/tracy -m clipbench.tt.run --test tracy_stage
##
## Same loop and same ranges as NVIDIA's nvtx_stats (deep_metrics.py), with Tracy zones in place
## of NVTX ranges: subtest > iter > subiter > decode / preprocess / encode / project.
## One view at a time, batch 1, no sync between views, one sync per frame.
##
## No CSVs: read the .tracy file in the Tracy GUI.
##   Host idle:   gaps between zones on the host track
##   Device idle: gaps between kernels on the device track (card waiting on the host)
##   Device late: the iter zone stretches at the frame sync while the card drains its queue
## encode/project zones only cover queuing the ops; the card's run time is on the device track.

import torch
import ttnn

from ...common import data
from .. import model_tt as model_lib

MAX_FRAMES = 2          # Profiler buffers are small and profiling is slow, so keep this short. Same as deep_metrics.py

## Tracy zone = NVTX range. Same names as deep_metrics.push()
def push(stage, views, view):
    ttnn.start_tracy_zone(__file__, stage, 0)
    return f'{stage}/v={views}/view={view}'


def pop(name):
    ttnn.stop_tracy_zone(name)


## One sub-test: every frame, at a single view count. timings is unused (run.py's shared signature)
def run(config, encoder, processor, projector, views_paths, views, timings):
    if model_lib.XLA:
        raise SystemExit('tracy_stage needs the ported ttnn model -- set run.xla: false')

    dtype = model_lib.resolve_dtype(config['run']['precision'])    # ttnn.bfloat16
    device = model_lib.tt_device.get_device()
    frames = list(data.iter_frames(views_paths))[:MAX_FRAMES]

    ## Empty the profiler buffers of anything left from warmup or the last sub-test
    model_lib.synchronize()
    ttnn.ReadDeviceProfiler(device)

    subtest = push('subtest', views, -1)                                    # Subtest: Zone Push

    with torch.inference_mode():
        for frame_paths in frames:
            it = push('iter', views, -1)                                    # Iter: Zone Push

            ## One sub-iteration per view, no synchronize between them
            for view, path in enumerate(frame_paths):
                sub = push('subiter', views, view)                          # Subiter: Zone Push

                ## Decode [CPU]
                z = push('decode', views, view)
                images = data.load_images([path])
                pop(z)

                ## Preprocess + upload [CPU -> TT], like NVIDIA's preprocess ending in .to(device)
                z = push('preprocess', views, view)
                pixel_values = data.preprocess(processor, images, 'cpu', torch.float32)
                pixel_values = model_lib.to_device(pixel_values, dtype)
                pop(z)

                ## Encode [TT]. Returns once the ops are queued
                z = push('encode', views, view)
                outputs = encoder(pixel_values, output_hidden_states=True)
                features = model_lib.select_image_features(outputs, config)
                pop(z)

                ## Project [TT]
                z = push('project', views, view)
                embeddings = projector(features)
                pop(z)

                pop(sub)                                                    # Subiter: Zone Pop

            ## Frame sync inside the iter zone, so 'iter' covers the device work too
            model_lib.synchronize()
            pop(it)                                                         # Iter: Zone Pop

            ## Empty the profiler buffers before the next frame. Outside the iter zone
            ttnn.ReadDeviceProfiler(device)

    model_lib.synchronize()
    pop(subtest)                                                            # Subtest: Zone Pop (includes the buffer reads)

    print('  pixel_values:', tuple(pixel_values.shape))
    print('  features    :', tuple(features.shape))
    print('  projected   :', tuple(embeddings.shape))

    return len(frames) * views
