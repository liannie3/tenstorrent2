"""Opening/closing the Tenstorrent card and the numeric settings shared by every TT-NN op in the model.

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


def sync(device):
    """TT equivalent of torch.cuda.synchronize(): block the host until every queued op on the card has finished."""
    ttnn.synchronize_device(device)


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


_FIDELITY = {"lofi": "LoFi", "hifi2": "HiFi2", "hifi4": "HiFi4"}
_WEIGHTS_DTYPE = {"bf16": "bfloat16", "bfp8": "bfloat8_b"}


class TtContext:
    """Device handle + numeric/memory settings that every TT module in tt_yolo.py reads.

    Layout rules used throughout the model (chosen to stay on the best-tested TT-NN paths):
      * convs, eltwise ops and matmuls take TILE tensors (row-major DRAM auto-slicing in conv2d can pick an oversized slice,
        tt-metal issue #51672)
      * max_pool2d, upsample and channel concats of non-32-aligned widths take ROW_MAJOR tensors
      * between ops every tensor is interleaved in `self.mem` (DRAM or L1); sharded op outputs are converted right away
    """

    def __init__(self, device, act_mem=None, weights_dtype=None, math_fidelity=None, fp32_dest_acc=None):
        self.device = device
        self.act_mem = act_mem or config.ACT_MEM
        if self.act_mem not in ("dram", "l1"):
            raise ValueError(f"act_mem must be 'dram' or 'l1', got {self.act_mem!r}")
        self.mem = ttnn.DRAM_MEMORY_CONFIG if self.act_mem == "dram" else ttnn.L1_MEMORY_CONFIG

        self.weights_dtype_name = weights_dtype or config.WEIGHTS_DTYPE
        self.weights_dtype = getattr(ttnn, _WEIGHTS_DTYPE[self.weights_dtype_name])
        #bfloat8_b weights must be handed to conv2d as fp32/bf16 row-major host tensors; bf16 weights go in as bf16
        self.host_weights_dtype = ttnn.float32 if self.weights_dtype_name == "bfp8" else ttnn.bfloat16

        self.math_fidelity_name = math_fidelity or config.MATH_FIDELITY
        fidelity = getattr(ttnn.MathFidelity, _FIDELITY[self.math_fidelity_name])
        fp32_acc = config.FP32_DEST_ACC if fp32_dest_acc is None else fp32_dest_acc
        self.fp32_dest_acc = bool(fp32_acc)
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=False,
        )
        #the box-decode matmuls are tiny, so they always run at full precision
        self.precise_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def sync(self):
        ttnn.synchronize_device(self.device)

    # ------------------------------------------------------------------ layout / placement helpers
    def interleaved(self, t):
        """Return `t` interleaved in the activation buffer (sharded outputs of conv/pool are moved here)."""
        if t.is_sharded() or t.memory_config().buffer_type != self.mem.buffer_type:
            return ttnn.to_memory_config(t, self.mem)
        return t

    def tile(self, t):
        """TILE layout in the activation buffer."""
        if t.layout != ttnn.TILE_LAYOUT:
            if self.act_mem == "l1":
                #tilizing straight into L1 can overflow for non-tile-aligned widths (noted in tt-metal's YOLO op tests),
                #so tilize in DRAM and then move the result to L1
                t = ttnn.to_layout(self.interleaved_dram(t), ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                t = ttnn.to_layout(self.interleaved(t), ttnn.TILE_LAYOUT, memory_config=self.mem)
        return self.interleaved(t)

    def row_major(self, t):
        """ROW_MAJOR layout in the activation buffer."""
        if t.layout != ttnn.ROW_MAJOR_LAYOUT:
            t = ttnn.to_layout(self.interleaved(t), ttnn.ROW_MAJOR_LAYOUT, memory_config=self.mem)
        return self.interleaved(t)

    @staticmethod
    def interleaved_dram(t):
        if t.is_sharded() or t.memory_config().buffer_type != ttnn.BufferType.DRAM:
            return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        return t

    def settings(self):
        """Settings recorded alongside every csv row."""
        return {
            "act_mem": self.act_mem,
            "weights_dtype": self.weights_dtype_name,
            "math_fidelity": self.math_fidelity_name + ("+fp32acc" if self.fp32_dest_acc else ""),
        }
