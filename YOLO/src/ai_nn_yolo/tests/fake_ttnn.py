"""CPU stand-in for `ttnn`, just enough of it to run the yolov8n port and the harness without a card.

Each op is done with torch on the CPU. It checks the things the port can get wrong on its own: tensor shapes, the
height/width/batch handed to every conv (the input must hold exactly N*H*W rows of in_channels), channel splits and concats,
the box decode, and use of a tensor after ttnn.deallocate. It can NOT tell whether the card accepts a sharding choice or
fits a feature map in L1; `main.py --setting verify` on the p150a does that.

FAKE_TTNN_DTYPE=fp32 (default) runs exact math, so the model must match ultralytics (PCC ~1.0).
FAKE_TTNN_DTYPE=bfp8 rounds conv weights and conv outputs to block-float-8 (16 values share an exponent, 7-bit mantissa)
to estimate the accuracy cost of the yolov8x demo's bfloat8_b setting (LoFi math is not emulated).
"""

import builtins
import os
import types

import torch
import torch.nn.functional as F

MODE = os.environ.get("FAKE_TTNN_DTYPE", "fp32")


def _q(t):
    """Round like the card would in the selected mode."""
    if MODE == "bf16":
        return t.to(torch.bfloat16).float()
    if MODE == "bfp8":
        shape = t.shape
        flat = t.reshape(-1, shape[-1]) if t.dim() > 1 else t.reshape(1, -1)
        pad = (-flat.shape[-1]) % 16
        x = F.pad(flat, (0, pad)).reshape(flat.shape[0], -1, 16)
        amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        step = torch.exp2(torch.floor(torch.log2(amax)) - 6)
        x = (torch.round(x / step).clamp(-127, 127) * step).reshape(flat.shape[0], -1)[:, : flat.shape[-1]]
        return x.reshape(shape)
    return t


# ------------------------------------------------------------------ constants / config objects
class _Enum:
    def __init__(self, *names):
        for n in names:
            setattr(self, n, n)


bfloat16, bfloat8_b, float32, uint32 = "bfloat16", "bfloat8_b", "float32", "uint32"
TILE_LAYOUT, ROW_MAJOR_LAYOUT = "TILE", "ROW_MAJOR"
MathFidelity = _Enum("LoFi", "HiFi2", "HiFi3", "HiFi4")
UnaryOpType = _Enum("SILU", "RELU")
TensorMemoryLayout = _Enum("INTERLEAVED", "HEIGHT_SHARDED", "BLOCK_SHARDED", "WIDTH_SHARDED")
ShardStrategy = _Enum("HEIGHT", "BLOCK", "WIDTH")
ShardOrientation = _Enum("ROW_MAJOR", "COL_MAJOR")
BufferType = _Enum("DRAM", "L1")
Conv2dL1FullSliceConfig = "L1Full"


class MemoryConfig:
    def __init__(self, kind):
        self.kind = kind
        self.buffer_type = BufferType.DRAM if kind == "dram" else BufferType.L1
        self.memory_layout = TensorMemoryLayout.HEIGHT_SHARDED if kind == "sharded" else TensorMemoryLayout.INTERLEAVED
        self.shard_spec = None

    def __repr__(self):
        return f"MemoryConfig({self.kind})"


DRAM_MEMORY_CONFIG = MemoryConfig("dram")
L1_MEMORY_CONFIG = MemoryConfig("l1")


class UnaryWithParam:
    def __init__(self, op, *params):
        self.op = op


class CoreCoord:
    def __init__(self, x, y):
        self.x, self.y = x, y


class CoreRange:
    def __init__(self, start, end):
        self.start, self.end = start, end


class CoreRangeSet:
    def __init__(self, ranges):
        self.ranges = list(ranges)


class CoreGrid:
    def __init__(self, x=8, y=8):
        self.x, self.y = x, y


class Conv2dConfig:
    def __init__(self, **kwargs):
        self.activation = None
        self.enable_weights_double_buffer = False
        self.act_block_h_override = 0
        for k, v in kwargs.items():
            setattr(self, k, v)


def init_device_compute_kernel_config(arch, **kwargs):
    return dict(kwargs)


def create_sharded_memory_config(*args, **kwargs):
    return MemoryConfig("sharded")


def create_sharded_memory_config_(*args, **kwargs):
    return MemoryConfig("sharded")


# ------------------------------------------------------------------ tensor
class Tensor:
    def __init__(self, t, layout=ROW_MAJOR_LAYOUT, mem=None, dtype=bfloat16):
        self._t = t
        self.layout = layout
        self._mem = mem or DRAM_MEMORY_CONFIG
        self.dtype = dtype

    @property
    def t(self):
        if self._t is None:
            raise RuntimeError("fake ttnn: tensor used after ttnn.deallocate")
        return self._t

    @property
    def shape(self):
        return tuple(self.t.shape)

    def memory_config(self):
        return self._mem

    def is_sharded(self):
        return self._mem.kind == "sharded"

    def volume(self):
        return self.t.numel()

    def _new(self, t, layout=None, mem=None):
        return Tensor(t, layout or self.layout, mem or self._mem, self.dtype)

    def __getitem__(self, idx):
        return self._new(self.t[idx].clone())

    def __add__(self, o):
        return self._new(self.t + (o.t if isinstance(o, Tensor) else o))

    def __sub__(self, o):
        return self._new(self.t - (o.t if isinstance(o, Tensor) else o))

    def __mul__(self, o):
        return self._new(self.t * (o.t if isinstance(o, Tensor) else o))

    __radd__ = __add__
    __rmul__ = __mul__


def _m(memory_config, x):
    return memory_config or x.memory_config()


# ------------------------------------------------------------------ device
class _Arch:
    def __str__(self):
        return "Arch.FAKE_BLACKHOLE"


class Device:
    def arch(self):
        return _Arch()

    def compute_with_storage_grid_size(self):
        return CoreCoord(12, 10)

    def get_num_devices(self):
        return 1

    def enable_program_cache(self):
        pass


def open_device(device_id=0, **kwargs):
    return Device()


def close_device(device):
    pass


def synchronize_device(device):
    pass


# ------------------------------------------------------------------ host <-> device
def from_torch(t, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None, spec=None):
    t = t.detach().float().clone()
    return Tensor(t, layout or ROW_MAJOR_LAYOUT, memory_config or DRAM_MEMORY_CONFIG, dtype or bfloat16)


def to_torch(x, mesh_composer=None, dtype=None):
    return x.t.clone()


# ------------------------------------------------------------------ data movement
def to_layout(x, layout, memory_config=None, dtype=None):
    return x._new(x.t.clone(), layout=layout, mem=memory_config)


def to_memory_config(x, memory_config, dtype=None):
    return x._new(x.t.clone(), mem=memory_config)


def sharded_to_interleaved(x, memory_config=None, output_dtype=None):
    return x._new(x.t.clone(), mem=memory_config or L1_MEMORY_CONFIG)


def interleaved_to_sharded(x, memory_config, *args, **kwargs):
    return x._new(x.t.clone(), mem=MemoryConfig("sharded"))


def reshard(x, memory_config, *args, **kwargs):
    return x._new(x.t.clone(), mem=MemoryConfig("sharded"))


def reallocate(x, memory_config=None):
    return x._new(x.t)


def deallocate(x, force=True):
    x._t = None


def clone(x, dtype=None, memory_config=None):
    return x._new(x.t.clone(), mem=memory_config)


def reshape(x, shape, memory_config=None):
    shape = tuple(shape)
    return x._new(x.t.reshape(shape).clone(), mem=memory_config)


def permute(x, dims, memory_config=None):
    return x._new(x.t.permute(*dims).contiguous(), mem=memory_config)


def pad(x, padding, value=0.0, memory_config=None):
    flat = []
    for before, after in reversed(padding):
        flat += [before, after]
    return x._new(F.pad(x.t, flat, value=value), mem=memory_config)


def concat(tensors, dim=0, memory_config=None):
    tensors = list(tensors)
    return tensors[0]._new(torch.cat([t.t for t in tensors], dim), mem=memory_config)


def slice(x, start, end, step=None, memory_config=None):
    for s, e, n in zip(start, end, x.shape):
        assert 0 <= s < e <= n, f"fake ttnn: slice {start}->{end} out of range for shape {x.shape}"
    idx = tuple(builtins.slice(s, e) for s, e in zip(start, end))
    return x._new(x.t[idx].clone(), mem=memory_config)



# ------------------------------------------------------------------ math
def add(a, b, memory_config=None, dtype=None):
    return a._new(a.t + (b.t if isinstance(b, Tensor) else b), mem=memory_config)


def multiply(a, b, memory_config=None, dtype=None):
    return a._new(a.t * (b.t if isinstance(b, Tensor) else b), mem=memory_config)


def div(a, b, memory_config=None):
    return a._new(a.t / (b.t if isinstance(b, Tensor) else b), mem=memory_config)


def sigmoid(x, memory_config=None):
    return x._new(torch.sigmoid(x.t), mem=memory_config)


def softmax(x, dim=-1, memory_config=None):
    return x._new(torch.softmax(x.t, dim), mem=memory_config)


def exp(x, memory_config=None):
    return x._new(torch.exp(x.t), mem=memory_config)


def conv2d(input_tensor, weight_tensor, in_channels, out_channels, device, bias_tensor, kernel_size, stride, padding,
           dilation, batch_size, input_height, input_width, conv_config=None, compute_config=None, groups=1,
           memory_config=None, return_weights_and_bias=False, return_output_dim=False, dtype=None, slice_config=None):
    x = input_tensor.t
    rows = batch_size * input_height * input_width
    assert x.numel() == rows * in_channels, (
        f"fake ttnn: conv2d input {tuple(x.shape)} doesn't hold batch {batch_size} x {input_height}x{input_width} "
        f"x {in_channels} channels")
    assert x.shape[-1] == in_channels, f"fake ttnn: conv2d input has {x.shape[-1]} channels, in_channels={in_channels}"
    x = x.reshape(batch_size, input_height, input_width, in_channels).permute(0, 3, 1, 2)

    w = weight_tensor.t if isinstance(weight_tensor, Tensor) else weight_tensor
    assert w.shape[0] == out_channels, f"fake ttnn: weight has {w.shape[0]} outputs, out_channels={out_channels}"
    if w.shape[1] * groups < in_channels:  #the stem's weights are 3-channel, its input is padded to 16
        w = F.pad(w, (0, 0, 0, 0, 0, in_channels // groups - w.shape[1]))
    assert w.shape[1] * groups == in_channels, f"fake ttnn: weight {tuple(w.shape)} vs in_channels={in_channels}"
    b = None
    if bias_tensor is not None:
        b = (bias_tensor.t if isinstance(bias_tensor, Tensor) else bias_tensor).reshape(-1)
        assert b.numel() == out_channels
    if MODE == "bfp8" and getattr(conv_config, "weights_dtype", None) == bfloat8_b:
        w = _q(w.reshape(w.shape[0], -1)).reshape(w.shape)

    y = F.conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
    act = getattr(conv_config, "activation", None)
    if act is not None and act.op == UnaryOpType.SILU:
        y = F.silu(y)
    elif act is not None and act.op == UnaryOpType.RELU:
        y = F.relu(y)
    n, c, oh, ow = y.shape
    y = y.permute(0, 2, 3, 1).reshape(1, 1, n * oh * ow, c)
    if dtype == bfloat8_b:
        y = _q(y)
    out = Tensor(y, getattr(conv_config, "output_layout", TILE_LAYOUT), MemoryConfig("sharded"), dtype or bfloat16)
    result = [out]
    if return_output_dim:
        result.append([oh, ow])
    if return_weights_and_bias:
        result.append([weight_tensor, bias_tensor])
    return result if len(result) > 1 else out


def max_pool2d(input_tensor, batch_size, input_h, input_w, channels, kernel_size, stride, padding, dilation,
               ceil_mode=False, memory_config=None, **kwargs):
    x = input_tensor.t
    assert x.numel() == batch_size * input_h * input_w * channels, "fake ttnn: max_pool2d shape mismatch"
    x = x.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)
    y = F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode=ceil_mode)
    n, c, oh, ow = y.shape
    return Tensor(y.permute(0, 2, 3, 1).reshape(1, 1, n * oh * ow, c), ROW_MAJOR_LAYOUT, MemoryConfig("sharded"))


def upsample(x, scale_factor, memory_config=None, mode="nearest"):
    t = x.t
    assert t.dim() == 4, "fake ttnn: upsample expects [N, H, W, C]"
    sh, sw = (scale_factor, scale_factor) if isinstance(scale_factor, int) else scale_factor
    y = t.repeat_interleave(sh, dim=1).repeat_interleave(sw, dim=2)
    return x._new(y, mem=memory_config)


def as_module():
    """The fake as a module object, for sys.modules["ttnn"]."""
    mod = types.ModuleType("ttnn")
    for k, v in globals().items():
        if not k.startswith("__"):
            setattr(mod, k, v)
    return mod
