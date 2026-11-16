import time
from collections import defaultdict
from contextlib import contextmanager
import torch_xla

import config

#nested dictionary to store timings of each stage's substage
timings = {s: defaultdict(list) for s in config.STAGES} #create dictionary to hold timing data


def reset():
    for s in config.STAGES:
        timings[s].clear()


def _card_sync():
    torch_xla.sync(wait = True) #the call tt's own tests use to wait for the card, instead of torch.cuda.synchronize()

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
