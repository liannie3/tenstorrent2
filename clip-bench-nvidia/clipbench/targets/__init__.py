"""Target registry.

Implementations are imported lazily so a CPU box never touches CUDA-only
modules -- and so a future tenstorrent target can import tt-forge without
every other platform paying for it.
"""

from .base import STAGES, Target


def _cpu():
    from .cpu import CpuTarget
    return CpuTarget


def _nvidia():
    from .nvidia import NvidiaTarget
    return NvidiaTarget


LOADERS = {'cpu': _cpu, 'nvidia': _nvidia}


def get_target(config):
    name = config['run']['target']
    try:
        loader = LOADERS[name]
    except KeyError:
        raise ValueError(f'Unknown target {name!r}, expected one of {sorted(LOADERS)}')
    return loader()(config)


__all__ = ['STAGES', 'Target', 'get_target', 'LOADERS']
