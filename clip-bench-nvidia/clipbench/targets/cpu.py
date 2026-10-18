"""CPU baseline: wall-clock stage timing with time.perf_counter.

This is the reference target and the one that produces the golden parity
tensor, so it stays simple and measures in-process -- no external profiler
needed to get a CSV.

It syncs at every stage boundary. On CPU there is nothing to drain, so that
costs a function call. On an accelerator the same pattern would serialise the
pipeline and destroy the CPU/GPU overlap you actually ship with, which is why
clipbench/targets/nvidia.py deliberately does not do it.
"""

import time
from contextlib import contextmanager

from .base import Target


class CpuTarget(Target):
    name = 'cpu'
    reports_in_process = True

    def __init__(self, config):
        super().__init__(config)
        self._spans = {}        # (batch_size, stage) -> [duration_ms, ...]

    @contextmanager
    def _span(self, batch_size, stage):
        self.sync()
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.sync()
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            self._spans.setdefault((batch_size, stage), []).append(elapsed_ms)

    @contextmanager
    def subtest(self, batch_size):
        with self._span(batch_size, 'subtest'):
            yield

    @contextmanager
    def iteration(self, batch_size, index):
        with self._span(batch_size, 'iter'):
            yield

    @contextmanager
    def stage(self, name, batch_size):
        with self._span(batch_size, name):
            yield

    def samples(self):
        return self._spans