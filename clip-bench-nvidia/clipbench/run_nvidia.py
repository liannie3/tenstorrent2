# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            run_nvidia.py
# CREATED:         10/19/2026
# REVISION:        2.0
# ======================================= #

"""NVIDIA runner. Times the host stages with perf_counter and the GPU stages
with CUDA events, and writes the same per-stage CSVs run_cpu.py writes.

    sudo nvpmodel -m 0        # MAXN -- do this before benchmarking
    sudo jetson_clocks        # pin clocks so sub-tests are comparable

    python -m clipbench.run_nvidia --profile jetson

Why not perf_counter for every stage like run_cpu.py: a CUDA call returns as
soon as its kernels are queued, so a host clock around `encoder(...)` measures
how long it took to *launch* the work, not how long the GPU spent on it. The
usual fix is torch.cuda.synchronize() after each stage, which drains the queue
four times an iteration and destroys exactly the CPU/GPU overlap the Orin
relies on in deployment.

A CUDA event is a timestamp written into the stream instead. Recording one does
not make the host wait, so the events sit around encode and project without
breaking the overlap, and the durations are read back after the sub-test's one
synchronize -- by which point every event has completed, so the read does not
block either. Host-bound stages (decode, preprocess) stay on perf_counter
because they do no GPU work for an event to time.

The NVTX ranges are still here but they no longer carry the measurements. They
cost a few hundred nanoseconds when no profiler is attached, and they are what
makes an occasional nsys capture readable when the question is *why* a stage is
slow (kernel breakdown, memcpy visibility, overlap on a visual timeline):

    nsys profile --trace=cuda,nvtx,osrt --output results/jetson/sweep \
         python -m clipbench.run_nvidia --profile jetson

    python -m clipbench.reports.nsys results/jetson/sweep.nsys-rep

That writes its own CSVs into a nsys/ subdirectory so it never overwrites the
event timings from this file. Note that a profiled run carries CUPTI overhead
on every kernel launch, so quote the unprofiled run for headline numbers.

Range names carry the batch size but deliberately NOT the iteration index.
`nsys stats --report nvtx_sum` groups by name, so 'encode/bs=4' stays one row
across every iteration of that sub-test; putting the index in the name would
make every range unique and defeat the aggregation.
"""

import argparse
import time
import warnings
from pathlib import Path

import torch
from torch.cuda import nvtx

from . import config as cfg
from . import data
from . import model as model_lib
from .reports.schema import write_manifest, write_stage_csvs


## Function checks to make sure NVTX is available
def _nvtx_available():
    try:
        nvtx.range_push('__probe__')
        nvtx.range_pop()
        return True
    except Exception:
        return False


NVTX_AVAILABLE = _nvtx_available()  # First check if NVTX is available


## The name format lives here only -- keep it in step with RANGE_PATTERN in
## clipbench/reports/nsys.py, which is what parses it back out of the capture
def push(stage, batch_size):
    if NVTX_AVAILABLE:
        nvtx.range_push(f'{stage}/bs={batch_size}')


def pop():
    if NVTX_AVAILABLE:
        nvtx.range_pop()


## One sub-test: the whole image set at a single batch size.
##
## Returns how many images ran and the wall-clock window the sub-test occupied.
## The window is what lets a tegrastats log be sliced per batch size -- see
## the note in main().
def run_subtest(config, encoder, processor, projector, paths, batch_size, timings):
    device = config['run']['device']
    dtype = model_lib.resolve_dtype(config['run']['precision'])
    max_batches = config['run'].get('max_batches')
    images_seen = 0
    batches_run = 0

    ## The events have to exist before the loop starts, so the batch count is
    ## worked out up front rather than discovered by iterating. Creating them
    ## here also keeps cudaEventCreate out of the timed region.
    num_batches = (len(paths) + batch_size - 1) // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)

    ## One (start, end) pair per iteration per GPU stage. enable_timing=True is
    ## what makes elapsed_time() legal on them later.
    encode_events = []
    project_events = []
    for i in range(num_batches):
        encode_events.append((torch.cuda.Event(enable_timing=True),
                              torch.cuda.Event(enable_timing=True)))
        project_events.append((torch.cuda.Event(enable_timing=True),
                               torch.cuda.Event(enable_timing=True)))

    ## decode + preprocess per iteration, kept so the 'iter' row can be put
    ## together after the GPU numbers come back
    host_iter_ms = []

    push('subtest', batch_size)
    subtest_start_unix = time.time()            # for lining up with tegrastats
    subtest_start = time.perf_counter_ns()      # for the duration itself

    ## inference_mode wraps the whole sub-test because nothing here ever trains
    with torch.inference_mode():
        for i, batch_paths in enumerate(data.iter_batches(paths, batch_size)):
            if i >= num_batches:
                break

            push('iter', batch_size)

            ## Pure CPU work -- file read plus JPEG decode in PIL -- so a host
            ## clock reads it correctly
            t0 = time.perf_counter_ns()
            push('decode', batch_size)
            images = data.load_images(batch_paths)
            pop()

            ## Host work too, ending in the H2D copy inside data.preprocess.
            ## That copy is from pageable memory, and CUDA syncs the stream
            ## before a pageable copy starts, so this stage quietly absorbs the
            ## previous iteration's GPU tail. The number is real but it is not
            ## purely preprocessing -- see the footnote at the end of the file.
            t1 = time.perf_counter_ns()
            push('preprocess', batch_size)
            pixel_values = data.preprocess(processor, images, device, dtype)
            pop()
            t2 = time.perf_counter_ns()

            ## GPU work. record() queues a timestamp and returns, so the host
            ## runs straight on into the next stage.
            push('encode', batch_size)
            encode_events[i][0].record()
            outputs = encoder(pixel_values, output_hidden_states=True)
            features = model_lib.select_image_features(outputs, config)
            encode_events[i][1].record()
            pop()

            push('project', batch_size)
            project_events[i][0].record()
            embeddings = projector(features)
            project_events[i][1].record()
            pop()

            pop()       # iter

            ## The host stages are known now and go straight into the buffer.
            ## The GPU stages cannot: elapsed_time() on an event that has not
            ## happened yet would block, which is the stall this file exists
            ## to avoid. They are read after the synchronize below.
            timings.append((batch_size, 'decode',     i, (t1 - t0) / 1e6))
            timings.append((batch_size, 'preprocess', i, (t2 - t1) / 1e6))
            host_iter_ms.append((t2 - t0) / 1e6)

            images_seen += len(batch_paths)
            batches_run += 1

    ## The only synchronize in the timed region. It sits inside the sub-test
    ## range so 'subtest' is a true wall-clock envelope including the GPU tail
    ## -- that envelope is what throughput is computed from. Once per batch
    ## size, so per-iteration overlap is untouched, and it stops one sub-test's
    ## trailing kernels landing inside the next sub-test's numbers.
    torch.cuda.synchronize()
    subtest_ms = (time.perf_counter_ns() - subtest_start) / 1e6
    subtest_end_unix = time.time()
    pop()               # subtest

    timings.append((batch_size, 'subtest', 0, subtest_ms))

    ## The stream is drained, so every event has completed and these reads
    ## return immediately. elapsed_time() gives milliseconds as a float.
    for i in range(batches_run):
        encode_ms = encode_events[i][0].elapsed_time(encode_events[i][1])
        project_ms = project_events[i][0].elapsed_time(project_events[i][1])
        timings.append((batch_size, 'encode',  i, encode_ms))
        timings.append((batch_size, 'project', i, project_ms))

        ## 'iter' here is the sum of the four stages, not a wall-clock reading.
        ## One iteration's host stages overlap the previous iteration's GPU
        ## stages, so there is no single interval that contains all four and
        ## nothing else. Use 'subtest' for throughput; this column is for
        ## comparing the stage mix against run_cpu.py.
        timings.append((batch_size, 'iter', i,
                        host_iter_ms[i] + encode_ms + project_ms))

    ## Shapes are the same every iteration, so the last batch describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    window = {
        'batch_size': batch_size,
        'start_unix': subtest_start_unix,
        'end_unix': subtest_end_unix,
    }
    return images_seen, window


def main(argv=None):
    ## Reads from command line (important to have --profile when running program)
    parser = argparse.ArgumentParser(description='CLIP encoder + MLP projector sweep, CUDA event timing')
    parser.add_argument('--profile', default='jetson', choices=cfg.available_profiles(),
                        help='config profile from configs/ (default: jetson)')
    args = parser.parse_args(argv)

    ## 1. Config
    config = cfg.load_config(args.profile)
    run = config['run']

    if run['target'] != 'nvidia':
        raise SystemExit(
            f"profile {args.profile!r} has run.target {run['target']!r} - run it with:\n"
            f'  python -m clipbench.run_cpu --profile {args.profile}'
        )

    ## The GPU stages are timed with CUDA events, so there is no CPU fallback
    ## path through this file. Refuse rather than crash halfway into a sweep.
    if not torch.cuda.is_available():
        raise SystemExit(
            'run_nvidia.py times the GPU stages with CUDA events and needs a working '
            'CUDA device, but torch.cuda.is_available() is False.\n'
            'For a CPU-only run use:\n'
            '  python -m clipbench.run_cpu --profile cpu'
        )

    if not NVTX_AVAILABLE:
        warnings.warn(
            'NVTX is unavailable, so the ranges will be no-ops and an nsys capture of '
            'this run will contain no stage annotations. The CSV timings are unaffected.',
            RuntimeWarning,
            stacklevel=2,
        )

    print(f"profile: {config['profile']}  device: {run['device']}  "
          f"precision: {run['precision']}")

    ## 2. Model pieces, built once and reused by every sub-test
    processor = model_lib.load_processor(config)                                 # from model.py
    encoder = model_lib.load_encoder(config)                                    # from model.py
    projector = model_lib.load_projector(config, encoder.config.hidden_size)    # from model.py

    ## 3. The image list, identical for every sub-test
    paths = data.load_paths(config)                                             # list of image paths
    print(f"found {len(paths)} images in {config['data']['path']}")
    if not paths:
        raise SystemExit('no images found - check data.path in configs/base.yaml')

    ## 4. The timing buffer. Appended to all through the sweep below, read once
    ## at step 6 and not before.
    timings = []        # (batch_size, stage, iter, duration_ms)
    windows = []        # one wall-clock window per sub-test, for tegrastats

    ## 5. One sub-test per batch size
    images_by_batch_size = {}
    for batch_size in run['batch_sizes']:
        print(f'\n=== sub-test: batch size {batch_size} over {len(paths)} images ===')
        images_seen, window = run_subtest(
            config, encoder, processor, projector, paths, batch_size, timings)
        images_by_batch_size[batch_size] = images_seen
        windows.append(window)
        print(f'  images      : {images_seen}')

    ## 6. Sweep is over, so now the buffer gets read and the files get written.
    ## The manifest records how the run was configured -- including the Jetson
    ## power mode, since DVFS makes two runs in different modes incomparable.
    results_dir = Path(config['output']['results_dir'])
    print('\nmanifest:', write_manifest(config, images_by_batch_size,
                                        results_dir / 'manifest.json',
                                        subtest_windows=windows))
    for stage, path in write_stage_csvs(timings, results_dir).items():
        print(f'{stage:>10}:', path)

    ## tegrastats samples at 100ms or slower, which is far too coarse to line up
    ## with a single stage, so it is not run from inside this process. Sub-tests
    ## last seconds though, and manifest.json now carries each one's wall-clock
    ## window, so a log recorded alongside the sweep can be sliced per batch
    ## size for power, clocks and thermals.
    interval = config.get('hardware', {}).get('tegrastats_interval_ms', 100)
    print('\nFor power and clock data, record a log alongside the next run:\n')
    print(f'  sudo tegrastats --interval {interval} '
          f'--logfile {results_dir / "tegrastats.txt"} &')
    print('  python -m clipbench.run_nvidia --profile '
          f'{config["profile"]}\n  sudo tegrastats --stop\n')
    print('Then slice it by the subtest_windows in manifest.json.')
    return 0


## Footnote on preprocess: data.preprocess ends in a .to(device) on a pageable
## tensor, and CUDA synchronizes the stream before a pageable H2D copy. So the
## preprocess number includes waiting for the previous iteration's encode to
## finish. Switching that copy to pinned memory with non_blocking=True would
## remove the wait and give a cleaner split -- but then preprocess becomes
## asynchronous too and needs its own event pair, like encode and project.
if __name__ == '__main__':
    raise SystemExit(main())
