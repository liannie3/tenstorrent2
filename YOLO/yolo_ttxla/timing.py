import time
from collections import defaultdict
from contextlib import contextmanager
import torch_xla
import torch_xla.core.xla_model as xm
import torch

import config


_barrier = None #force to wait to return before card has finished all queued compiled graph work
#nested dictionary to store timings of each stage's substage
timings = {s: defaultdict(list) for s in config.STAGES} #create dictionary to hold timing data


def reset():
    for s in config.STAGES:
        timings[s].clear()


def _card_sync():
    #original not working as intended
    #torch_xla.sync(wait = True)
    #xm.wait_device_ops()

    #NOTE: BELOW addition of barrier to only capture time once all queued compiled graph work is finished and not just when it has been submitted to the card
    #torch_xla.sync likely does this as it appears to return early, before compiled graph work is actually done, after testing

    #following run_inference() function from tt-xla repo examples/pytorch/mnist_performant.py
    #makes the host wait until the card has finished all its queued work by asking for a tiny result
    #that it can only produce after finishing all work ahead of it
    #NOTE: includes up to ~0.6ms of barrier overhead which is signficant for small batch size runs, potentially want to subtract from normalization, copy back , and initial host to card copy.
    global _barrier
    torch_xla.sync(wait = True) #submit any already recorded work so it is queued before barrier func
    if _barrier is None: #only empty on first run
        _barrier = torch.zeros(1, device = config.DEVICE) #create 1 element tensor holding 0 on the card
        torch_xla.sync(wait = True) #forces the barrier to actually be created and filled before being used below
    (_barrier + 1).cpu() #tiny new piece of work queued after all earlier work, then read back to the host

#used during wall timing runs to synchronize CPU to wait for GPU work
def _sync(device):
    #substep timing is purely async so never want to invoke a CPU sync
    if not config.SUBSTEP_TIMING and not config.HARDWARE_METRICS: #TT CHANGE: no device type check, this harness always runs on the card
        _card_sync()

#timer helper
#TODO: add markers for TT's profiler, Tracy
@contextmanager
def timed(store, name, device, gpu_work = True):
    if not config.SUBSTEP_TIMING or config.HARDWARE_METRICS: #wall timing uses perf counters and sync so timed unapplicable
        yield
        return
    if gpu_work:
        _card_sync()
    t0 = time.perf_counter()
    yield  #exit code below
    if gpu_work:
        _card_sync()
    store[name].append((time.perf_counter() - t0 ) * 1000) #convert to ms timing and store in dictionary at associated key
