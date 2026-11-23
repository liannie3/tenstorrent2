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
import os
import torch
from transformers import CLIPVisionModel

## Processor and projector init are platform-neutral -- reusing them guarantees
## identical preprocessing and identical projector weights for the same seed
from ..nvidia.model_nvidia import MLPProjector, load_processor  # noqa: F401  (load_processor re-exported)

## ttnn may not be installed in the TT-XLA venv. torch_xla is imported in open_device,
## after TT_RUNTIME_TRACE_REGION_SIZE is set.
try:
    import ttnn
    from .tt_clip import device as tt_device
    from .tt_clip.vision import VisionTransformer
    from .tt_clip.weights import convert_model_to_ttnn
except ImportError:
    ttnn = None

## Initialization of XLA
# XLA originally set as False but will be updated depending on what is set in tt.yaml
XLA = False
xla_device = None


## Only precision allowed is bf16
def resolve_dtype(precision):
    if precision != "bf16":                                                       # Check to see if the data is bf16
        raise ValueError(f"Unknown precision {precision!r}, expected 'bf16'")     # Throws an error if it's not
    return torch.bfloat16 if XLA else ttnn.bfloat16                               # Returns torch bf16 if XLA is on, ttnn bf16 if off



## Must be open before load_encoder / load_projector -- weights go straight to the card
# CUDA doesn't have to open and close device because it automatically does it for you
# Only one process can hold a TT device at a time
def open_device(config):
    global XLA                    # Set globals
    global xla_device 

    run = config['run']           # Reads the run parameters from tt.yaml
    XLA = run.get('xla', False)   # Returns True or False depending on if XLA is on or off

    ##############################################
    ##                  XLA On                  ##
    ##############################################
    if XLA:                       # XLA is set to True
        if run.get('trace'):      # Check whether Trace is on or off (Tenstorrent's equivalent to CUDA Graphs)
            os.environ.setdefault('TT_RUNTIME_TRACE_REGION_SIZE', str(run['trace_region_size']))  # Sets aside a region of the DRAM to hold a recorded trace

        ## Import required libraries for XLA
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        xr.set_device_type('TT')                                        # Set-up XLA once and point it to the Tenstorrent card
        options = {'optimization_level': run.get('xla_opt_level', 1)}   # By default, torch-xla runs on CPU. This line switches to the PJRT pugin so
                                                                        # tt-xla goes to the card (0 none, 1 recommended, 2 max SRAM)

        if run.get('trace'):                                            # Check if trace is on or off
            options['enable_trace'] = 'true'                            # Passed to the options as a string

        torch_xla.set_custom_compile_options(options)                   # This passes the options we just set to tt-xla

        xla_device = xm.xla_device()                                    # Gets the handle for the device
        return xla_device


    ##############################################
    ##                  XLA Off                 ##
    ##############################################
    ## This section runs if XLA is turned off
    if ttnn is None:                                                                         # Check if ttnn is installed
        raise SystemExit('run.xla is false but ttnn is not installed in this environment')   # If not, throw an error
    
    tt_device.open_ttnn(device_id=run['device_id'],
                        l1_small_size=run['l1_small_size'],               # how many bytes of each core's on-chip memory ttnn sets aside
                        trace_region_size=run['trace_region_size'])       # equivalent to CUDA Graphs
    
    return tt_device.get_device()



## Closes Device
# Finishes any queued work and frees everything allocated on the card (weights, activations, traces)
# Releases the card
def close_device():
    if XLA:
        synchronize()
    else:
        tt_device.close_ttnn()

## Block until all queued device work is done
def synchronize():
    if XLA:
        import torch_xla
        import torch_xla.core.xla_model as xm

        torch_xla.sync()                         # flush the pending graph (torch-xla is lazy)
        xm.wait_device_ops()
    else:
        ttnn.synchronize_device(tt_device.get_device())


## The .to(device) at the end of data.preprocess.
## Call data.preprocess with device='cpu', dtype=torch.float32, then this.
def to_device(pixel_values, dtype):
    if XLA:
        return pixel_values.to(dtype).to(xla_device)
    return ttnn.from_torch(pixel_values, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                           device=tt_device.get_device())


## CLIP encoder, frozen. Runs all 24 layers; call it with output_hidden_states=True
## and select_image_features picks hidden_states[layer_index], same as NVIDIA.
def load_encoder(config):
    resolve_dtype(config['run']['precision'])     # fail early if precision isn't bf16
    hf_model = CLIPVisionModel.from_pretrained(config['model']['name']).eval()

    if XLA:
        ## Stock HF model, the same one NVIDIA runs, compiled for the card
        hf_model = hf_model.to(torch.bfloat16).requires_grad_(False)
        hf_model.vision_width = hf_model.config.hidden_size          # 1024, read by run.py
        hf_model.compile(backend='tt')                               # compiles on the first call
        return hf_model.to(xla_device)

    ## HF model only used as the source of the weights
    num_layers = hf_model.config.num_hidden_layers                   # 24 for ViT-L
    return VisionTransformer(convert_model_to_ttnn(hf_model.state_dict()), num_vision_layers=num_layers)


class TTMLPProjector:
    def __init__(self, state_dict):                   # Stores 4 Tensors (Weight and Bias for each of the 2 linear layers)
        self.fc1_weight = state_dict['fc1.weight']    # [2048, 1024] Maps each 1024-number CLIP vector to 2048 numbers
        self.fc1_bias = state_dict['fc1.bias']        # [2048] A constant added to each of the 2048 outputs
        self.fc2_weight = state_dict['fc2.weight']    # [2048, 2048] Mixes the 2048 numbers into the final 2048-number embedding
        self.fc2_bias = state_dict['fc2.bias']        # Constant added to final outputs

    def forward(self, image_features):
        x = ttnn.linear(image_features, self.fc1_weight, bias=self.fc1_bias, transpose_b=True)
        x = ttnn.gelu(x, fast_and_approximate_mode=False)                                        # exact GELU, same as nn.GELU()
        return ttnn.linear(x, self.fc2_weight, bias=self.fc2_bias, transpose_b=True)

    __call__ = forward


## Built in torch with the same seed as NVIDIA
def load_projector(config, in_dim):
    torch.manual_seed(config['run']['seed'])
    projector = MLPProjector(in_dim, config['model']['projector_out_dim'])

    if XLA:
        projector = projector.to(torch.bfloat16).eval().requires_grad_(False)
        projector.compile(backend='tt')
        return projector.to(xla_device)

    return TTMLPProjector(convert_model_to_ttnn(projector.state_dict()))


def select_image_features(outputs, config):
    features = outputs.hidden_states[config['model']['layer_index']]
    if not config['model']['keep_cls']:
        if XLA:
            features = features[:, 1:, :]
        else:
            batch, seq, hidden = features.shape
            features = ttnn.slice(features, [0, 1, 0], [batch, seq, hidden])
    return features
