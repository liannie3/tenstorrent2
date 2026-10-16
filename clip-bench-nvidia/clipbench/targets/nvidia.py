"""NVIDIA target: NVTX annotation, no in-process timing.

NVTX ranges are labels, not timers. `torch.cuda.nvtx.range_push` costs a few
hundred nanoseconds when no profiler is attached and hands no duration back to
Python. So this target annotates the pipeline and otherwise stays out of the
way: there is no torch.cuda.synchronize() at stage boundaries.

That absence is the point. CUDA is asynchronous -- the encoder call returns once
its kernels are queued -- so a wall clock can only read it by draining the queue
first, four times an iteration, which forbids exactly the CPU/GPU overlap the
Orin will rely on in deployment. NVTX needs no drain, and Nsight Systems
reconstructs the real timeline afterwards.

Record a run:

    nsys profile --trace=cuda,nvtx,osrt \\
         --output results/jetson/sweep \\
         python -m clipbench.runner --profile jetson

Reading the capture back is deliberately not implemented here.

Range names carry the batch size but deliberately NOT the iteration index.
`nsys stats --report nvtx_sum` groups by name, so 'encode/bs=4' collapses into a
single row across every iteration of that sub-test. Putting the index in the
name would make every range unique and defeat the aggregation entirely.
"""

import warnings
from contextlib import contextmanager

import torch
from torch.cuda import nvtx

from .base import Target


## torch ships NVTX stubs that raise RuntimeError on a CPU-only wheel. Probe
## once at import so the module stays importable on a dev box and under pytest
## instead of exploding at the first range.
def _nvtx_available():
    try:
        nvtx.range_push('__probe__')
        nvtx.range_pop()
        return True
    except Exception:
        return False


NVTX_AVAILABLE = _nvtx_available()


## Keep in step with RANGE_PATTERN in clipbench/reports/nsys.py
def range_name(stage, batch_size):
    return f'{stage}/bs={batch_size}'


class NvidiaTarget(Target):
    name = 'nvidia'

    ## nsys holds the durations; clipbench.reports.nsys turns them into rows
    reports_in_process = False

    def __init__(self, config):
        super().__init__(config)
        if not NVTX_AVAILABLE:
            warnings.warn(
                'NVTX is unavailable (CPU-only torch build?). Ranges will be no-ops and '
                'an nsys capture of this run will contain no stage annotations.',
                RuntimeWarning,
                stacklevel=2,
            )

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def _range(self, name):
        if not NVTX_AVAILABLE:
            yield
            return
        nvtx.range_push(name)
        try:
            yield
        finally:
            nvtx.range_pop()

    @contextmanager
    def subtest(self, batch_size):
        with self._range(range_name('subtest', batch_size)):
            try:
                yield
            finally:
                ## The one sync in the whole run. It sits inside the range so
                ## 'subtest/bs=N' is a true wall-clock envelope of the sub-test
                ## including the GPU tail -- that envelope is what throughput is
                ## computed from. Once per batch size, so per-iteration overlap
                ## is untouched, and it stops one sub-test's trailing kernels
                ## landing inside the next sub-test's ranges.
                self.sync()

    @contextmanager
    def iteration(self, batch_size, index):
        with self._range(range_name('iter', batch_size)):
            yield

    @contextmanager
    def stage(self, name, batch_size):
        with self._range(range_name(name, batch_size)):
            yield
