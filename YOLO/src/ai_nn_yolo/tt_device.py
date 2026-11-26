"""Opening/closing the Tenstorrent card, plus the settings recorded alongside every csv row.

Replaces the bits of the CUDA version that were spread across `config.DEVICE = "cuda"`, `.to(DEVICE)` and
`torch.cuda.synchronize()`.
"""

from contextlib import contextmanager

import ttnn

import config


@contextmanager
def open_device(device_id=None):
    """Open the card, enable the program cache, and always close it again (a device left open blocks the next run)."""
    device = ttnn.open_device(
        device_id=config.DEVICE_ID if device_id is None else device_id,
        l1_small_size=config.L1_SMALL_SIZE,
        trace_region_size=config.TRACE_REGION_SIZE,
        num_command_queues=config.NUM_COMMAND_QUEUES,
    )
    try:
        #the program cache stores compiled kernels + runtime args per op shape, so repeated batches skip host-side setup.
        #(newer tt-metal enables it by default; the call is kept for older releases)
        if hasattr(device, "enable_program_cache"):
            device.enable_program_cache()
        yield device
    finally:
        ttnn.close_device(device)


def describe(device):
    """Short human-readable description, e.g. 'blackhole 12x10 grid'."""
    try:
        arch = str(device.arch()).split(".")[-1].lower()
    except Exception:
        arch = "tt"
    try:
        grid = device.compute_with_storage_grid_size()
        return f"{arch} {grid.x}x{grid.y} grid"
    except Exception:
        return arch


class TtContext:
    """Device handle + the settings every csv row records.

    NOTE: the yolov8x-derived model places its own tensors (L1 height/block sharded between ops, the three detect inputs
          cloned to DRAM), so there is no activation-memory switch here anymore
    """

    def __init__(self, device):
        self.device = device

    def sync(self):
        ttnn.synchronize_device(self.device)

    def settings(self):
        """Settings recorded alongside every csv row (same columns as before, so the csv header doesn't change)."""
        return {
            "act_mem": "l1_sharded",
            "weights_dtype": config.WEIGHTS_DTYPE,
            "math_fidelity": config.MATH_FIDELITY,
        }
