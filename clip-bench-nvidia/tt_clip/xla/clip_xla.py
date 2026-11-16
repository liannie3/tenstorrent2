"""TT-XLA CLIP vision encoder: the stock HuggingFace model, compiled for the Tenstorrent card.

Counterpart of the hand-ported TT-NN model in the parent tt_clip/ folder. No ops are
rewritten here -- TT-XLA traces the PyTorch model through torch-xla to StableHLO and
TT-MLIR compiles it. Output matches tt_clip's TTClipVisionModel: the hidden state after
the first 23 of 24 layers with CLS dropped, [batch, 576, 1024].

API follows tenstorrent/tt-xla examples/pytorch/mnist_performant.py:
  xr.set_device_type("TT") -> set_custom_compile_options -> model.compile(backend="tt")
  -> xm.xla_device() -> model.to(device)
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import CLIPVisionModel


class CLIPVisionFeatures(torch.nn.Module):
    """CLIPVisionModel cut to `num_layers` encoder layers, returning patch tokens only.

    HF's last_hidden_state is the encoder output before post_layernorm (that norm only
    touches the pooled CLS output), so after truncating to 23 layers it equals the full
    model's hidden_states[-2] -- the same tensor tt_clip/test.py compares against.
    Truncating, instead of asking for output_hidden_states=True, keeps the 24th layer
    and the other 24 hidden states out of the compiled graph and off the readback.
    """

    def __init__(self, hf_model: CLIPVisionModel, num_layers: int):
        super().__init__()
        encoder = hf_model.vision_model.encoder
        encoder.layers = encoder.layers[:num_layers]
        self.vision_model = hf_model.vision_model

    def forward(self, pixel_values):
        hidden = self.vision_model(pixel_values=pixel_values).last_hidden_state
        return hidden[:, 1:, :]                     # drop CLS -> [batch, 576, 1024]


def build(hf_model: CLIPVisionModel, num_layers: int, optimization_level: int = 1,
          enable_trace: bool = False):
    """Wrap, cast to bf16, compile for TT and move to the device. Returns (model, device).

    Call xr.set_device_type("TT") first (open_tt does). For enable_trace=True the
    TT_RUNTIME_TRACE_REGION_SIZE env var must be set before torch_xla is imported.
    """
    model = CLIPVisionFeatures(hf_model, num_layers).eval()
    model = model.to(dtype=torch.bfloat16)          # bf16, like the TT-NN port

    options = {"optimization_level": optimization_level}   # 0 none, 1 recommended, 2 max SRAM
    if enable_trace:
        options["enable_trace"] = "true"
    torch_xla.set_custom_compile_options(options)

    model.compile(backend="tt")                     # compiles lazily on the first forward
    device = xm.xla_device()
    return model.to(device), device


def open_tt():
    """torch_xla defaults to the CPU device; point it at the Tenstorrent card."""
    xr.set_device_type("TT")


def wait():
    """Block until queued device work finishes -- the ttnn.synchronize_device of torch-xla."""
    xm.wait_device_ops()
