import time
from collections import defaultdict
from contextlib import contextmanager

import ttnn

import config

#nested dictionary to store timings of each stage's substage
timings = {s: defaultdict(list) for s in config.STAGES} #create dictionary to hold timing data
current = {"step": None} #substep that is running right now, so a failure can say where it happened


def reset():
    for s in config.STAGES:
        timings[s].clear()


#used during wall timing runs to make the CPU wait for the card between stages (torch.cuda.synchronize in the CUDA version)
def _sync(device):
    #substep timing already synchronizes around every device substep, and the end-to-end pass must not sync at all
    if config.SYNC_STAGES and not config.SUBSTEP_TIMING:
        ttnn.synchronize_device(device)


#Tracy markers: the tt-metal equivalent of nvtx.range_push/range_pop (a no-op unless TRACY_MARKERS is on)
@contextmanager
def tracy_zone(name):
    start = getattr(ttnn, "start_tracy_zone", None) if config.TRACY_MARKERS else None
    if start is None:
        yield
        return
    start(__file__, name, 0)
    try:
        yield
    finally:
        ttnn.stop_tracy_zone(name)


def signpost(name):
    """Mark a point in the ops perf report (tracy's TT_SIGNPOST), e.g. the start of a stage."""
    if config.TRACY_MARKERS and hasattr(ttnn, "tracy_message"):
        ttnn.tracy_message(f"`TT_SIGNPOST: {name}`")


#timer helper
#NOTE: the CUDA version timed GPU substeps with CUDA events (device timestamps, no host sync). tt-metal events carry no
#      timestamps, so a device substep is timed here as: drain the queue, perf_counter, run, drain again, perf_counter.
#      That includes one host<->device round trip per substep, which is why the substep pass sums a bit above the wall pass.
#      Pure on-device kernel times come from the tt-metal profiler (Tracy), which is a later step.
@contextmanager
def timed(store, name, device, device_work=True):
    current["step"] = name
    if not config.SUBSTEP_TIMING:
        with tracy_zone(name):
            yield
        return
    if device_work:
        ttnn.synchronize_device(device) #start from an idle card so earlier queued work isn't billed to this substep
    t0 = time.perf_counter()
    with tracy_zone(name):
        yield
    if device_work:
        ttnn.synchronize_device(device) #wait for this substep's ops to finish on the card
    store[name].append((time.perf_counter() - t0) * 1000) #convert to ms timing and store in dictionary at associated key
