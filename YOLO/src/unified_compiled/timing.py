import time
from collections import defaultdict
from contextlib import contextmanager
import torch

import config

#nested dictionary to store timings of each stage's substage
timings = {s: defaultdict(list) for s in config.STAGES} #create dictionary to hold timing data
async_timings = [] #GPU event pairs that are waiting to be read

def reset():
    for s in config.STAGES:
        timings[s].clear()
    async_timings.clear()

#used during wall timing runs to synchronize CPU to wait for GPU work
def _sync(device):
    #substep timing is purely async so never want to invoke a CPU sync 
    if torch.device(device).type == "cuda" and not config.SUBSTEP_TIMING and not config.HARDWARE_METRICS:
        torch.cuda.synchronize()

#timer helper
#TODO: WILL HAVE TO ADJUST OR CREATE NEW VERSION FOR TT
@contextmanager
def timed(store, name, device, gpu_work = True):
    if not config.SUBSTEP_TIMING or config.HARDWARE_METRICS: #wall timing uses perf counters and sync so timed unapplicable
        torch.cuda.nvtx.range_push(name) #add nvtx marker to record substage
        yield
        torch.cuda.nvtx.range_pop() #pop nvtx marker and record timestamp associated with this stage's work
        return
    if torch.device(device).type == "cuda" and gpu_work:
        start = torch.cuda.Event(enable_timing = True) #marker/timestamp for event at start of substage
        end = torch.cuda.Event(enable_timing = True) #marker/timestamp for event at end of substage
        start.record()  # record timestamp when gpu reaches this event (i.e. start of substage) in def stream
        torch.cuda.nvtx.range_push(name) #add nvtx marker to record substage
        yield
        torch.cuda.nvtx.range_pop() #pop nvtx marker and record timestamp associated with this stage's work
        end.record() # record timestamp when gpu reaches this event (i.e. end of substage) in def stream
        end.query() #pushes from the driver's buffered commands to the GPU now, so the endmarker of events is promptly timestamped
        async_timings.append((store, name, start, end)) #store timing data to be processed async after run
    else:
        #NOTE: no need to synchronize for steps that occur before transfer from host to device since no GPU events for the CPU to wait for
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push(name) #add nvtx marker to record substage
        yield  #exit code below 
        torch.cuda.nvtx.range_pop() #pop nvtx marker and record timestamp associated with this stage's work
        store[name].append((time.perf_counter() - t0 ) * 1000) #convert to ms timing and store in dictionary at associated key

def process_timings(): 
    #reads all pending GPU events achronously at end of run
    if async_timings: #if events are pending
        torch.cuda.synchronize() #force CPU to wait for GPU, this occurs following run so no worries about sync vs. async
        for store, name, start, end in async_timings: #store difference of start and end event for each substage in corresponding dictionary
            store[name].append(start.elapsed_time(end)) 
        async_timings.clear() #empty for subsequent runs

