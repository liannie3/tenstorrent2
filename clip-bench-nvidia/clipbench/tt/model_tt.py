# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            model_tt.py
# CREATED:         11/13/2026
# REVISION:        1.0
# ======================================= #

"""Model definitions, Tenstorrent side.
Same functions as nvidia/model_nvidia.py so the two runners read the same. The
differences: weights are converted to ttnn, the encoder stops at layer_index
instead of returning every hidden state, and pixels need an explicit upload.
"""

import torch
import ttnn
from transformers import CLIPVisionModel

## Processor and projector init are platform-neutral -- reusing them guarantees
## identical preprocessing and identical projector weights for the same seed
from ..nvidia.model_nvidia import MLPProjector, load_processor  # noqa: F401  (load_processor re-exported)

from .tt_clip import device as tt_device
from .tt_clip.vision import VisionTransformer
from .tt_clip.weights import convert_model_to_ttnn

## Allowed run types for model. tt_clip/weights.py casts everything to bf16.
DTYPES = {'bf16': ttnn.bfloat16}


## Checks if precision is allowed (set by DTYPES)
def resolve_dtype(precision):
    try:
        return DTYPES[precision]
    except KeyError:
        raise ValueError(f'Unknown precision {precision!r}, expected one of {tuple(DTYPES)}')


## Must be open before load_encoder / load_projector -- weights go straight to the card
def open_device(config):
    run = config['run']
    tt_device.open_ttnn(device_id=run['device_id'],
                        l1_small_size=run['l1_small_size'],
                        trace_region_size=run['trace_region_size'])
    return tt_device.get_device()


def close_device():
    tt_device.close_ttnn()


## The ttnn equivalent of the .to(device) at the end of data.preprocess.
## Call data.preprocess with device='cpu', dtype=torch.float32, then this.
def to_device(pixel_values, dtype):
    return ttnn.from_torch(pixel_values, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=tt_device.get_device())


## hidden_states has num_layers + 1 entries (embeddings first), so layer_index -2
## on a 24-layer model is the output of layer 23. The TT encoder just runs that many.
def _num_layers(layer_index, total_layers):
    return layer_index if layer_index >= 0 else total_layers + 1 + layer_index


## CLIP encoder, frozen. Returns hidden_states[layer_index] directly, CLS included.
def load_encoder(config):
    resolve_dtype(config['run']['precision'])
    hf_model = CLIPVisionModel.from_pretrained(config['model']['name']).eval()
    num_layers = _num_layers(config['model']['layer_index'], hf_model.config.num_hidden_layers)
    return VisionTransformer(convert_model_to_ttnn(hf_model.state_dict()), num_vision_layers=num_layers)


class TTMLPProjector:
    def __init__(self, state_dict):
        self.fc1_weight = state_dict['fc1.weight']
        self.fc1_bias = state_dict['fc1.bias']
        self.fc2_weight = state_dict['fc2.weight']
        self.fc2_bias = state_dict['fc2.bias']

    def forward(self, image_features):
        x = ttnn.linear(image_features, self.fc1_weight, bias=self.fc1_bias, transpose_b=True)
        x = ttnn.gelu(x, fast_and_approximate_mode=False)    # exact erf GELU, same as nn.GELU()
        return ttnn.linear(x, self.fc2_weight, bias=self.fc2_bias, transpose_b=True)

    __call__ = forward


## Built in torch with the same seed as NVIDIA, then converted
def load_projector(config, in_dim):
    run = config['run']
    torch.manual_seed(run['seed'])
    projector = MLPProjector(in_dim, config['model']['projector_out_dim'])
    return TTMLPProjector(convert_model_to_ttnn(projector.state_dict()))


## Layer choice already happened in load_encoder; only the CLS drop is left
def select_image_features(features, config):
    if not config['model']['keep_cls']:
        batch, seq, hidden = features.shape
        features = ttnn.slice(features, [0, 1, 0], [batch, seq, hidden])
    return features
