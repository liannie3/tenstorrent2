# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            model.py
# CREATED:         10/15/2026
# REVISION:        1.0
# ======================================= #

"""Model definitions.
Nothing here knows what hardware it is on -- device placement goes through the
target so that CPU, Jetson and anything added later build the same graph.
"""

import torch
import torch.nn as nn
import transformers
from transformers import CLIPVisionModel

## Allowed run types for model
DTYPES = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}

## `CLIPImageProcessor` silently picks a torchvision backend when torchvision is
## installed and a PIL one when it is not. That would change preprocessing
## between the dev box and the Orin and quietly invalidate the golden parity
## tensor, so the backend comes from config instead of from the environment.
PROCESSOR_BACKENDS = {'pil': 'CLIPImageProcessorPil', 'torchvision': 'CLIPImageProcessor'}


## Checks if precision is allowed (set by DTYPES)
def resolve_dtype(precision):
    try:
        return DTYPES[precision]
    except KeyError:
        raise ValueError(f'Unknown precision {precision!r}, expected one of {tuple(DTYPES)}')


def load_processor(config):
    backend = config['model'].get('image_processor', 'pil')
    try:
        class_name = PROCESSOR_BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f'Unknown image_processor {backend!r}, expected one of {tuple(PROCESSOR_BACKENDS)}'
        )
    return getattr(transformers, class_name).from_pretrained(config['model']['name'])


## CLIP encoder, frozen
def load_encoder(config, target):
    model = CLIPVisionModel.from_pretrained(
        config['model']['name'],
        dtype=resolve_dtype(config['run']['precision']),
    )
    model = target.prepare(model)
    model.eval()
    model.requires_grad_(False)
    return model


class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, image_features):
        return self.fc2(self.act(self.fc1(image_features)))


def load_projector(config, target, in_dim):
    run = config['run']
    torch.manual_seed(run['seed'])
    projector = MLPProjector(in_dim, config['model']['projector_out_dim'])
    projector = target.prepare(projector, dtype=resolve_dtype(run['precision']))
    projector.eval()
    projector.requires_grad_(False)
    return projector


def select_image_features(outputs, config):
    features = outputs.hidden_states[config['model']['layer_index']]
    if not config['model']['keep_cls']:
        features = features[:, 1:, :]
    return features
