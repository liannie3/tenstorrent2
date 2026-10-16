"""The one seam between the shared benchmark and a specific piece of hardware.

Everything platform-dependent lives behind this interface -- device placement,
synchronisation, and how a stage gets timed -- so clipbench/runner.py holds
exactly one sweep loop and never branches on the device.

Adding a target means implementing this class, not copying the loop.
"""

from contextlib import contextmanager

## The stages annotated inside each iteration, in pipeline order
STAGES = ('decode', 'preprocess', 'encode', 'project')


class Target:
    name = 'base'

    ## False when durations only exist inside an external profiler's capture.
    ## The runner uses this to decide between writing a CSV and pointing you at
    ## a report reader.
    reports_in_process = True

    def __init__(self, config):
        self.config = config
        self.run = config['run']
        self.device = self.run['device']

    ## Module placement. A future accelerator may compile here instead of
    ## moving, which is why the runner never calls .to() itself.
    def prepare(self, module, dtype=None):
        if dtype is None:
            return module.to(self.device)
        return module.to(device=self.device, dtype=dtype)

    def to_device(self, tensor, dtype=None):
        return tensor.to(device=self.device, dtype=dtype)

    def sync(self):
        pass

    ## One batch size, over the whole image set
    @contextmanager
    def subtest(self, batch_size):
        yield

    ## One batch
    @contextmanager
    def iteration(self, batch_size, index):
        yield

    ## One stage within a batch; name is one of STAGES
    @contextmanager
    def stage(self, name, batch_size):
        yield

    ## Canonical rows (see clipbench.reports.schema), or None when an external
    ## profiler holds the timings and a reader has to produce them instead
    def rows(self, images_by_batch_size):
        return None
