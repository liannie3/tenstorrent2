import torch
import torch.nn as nn
from transformers import CLIPImageProcessorPil, CLIPVisionModel

import harness.config as cfg
import harness.data as data

## Allowed run types for model
DTYPES = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}

## Checks if precision is allowed (set by DTYPES)
def resolve_dtype(precision):
    try:
        return DTYPES[precision]
    except KeyError:
        raise ValueError(f"Unknown precision {precision!r}, expected one of {tuple(DTYPES)}")

## CLIP Encoder
def load_model_and_processor(config):
    run = config['run']                    
    name = config['model']['name']        # Assigns model defined in YAML

    model = CLIPVisionModel.from_pretrained(name, dtype=resolve_dtype(run['precision']))
    model.to(run['device'])
    model.eval()
    model.requires_grad_(False)

    processor = CLIPImageProcessorPil.from_pretrained(name)
    
    return model, processor

class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, image_features):
        return self.fc2(self.act(self.fc1(image_features)))


def load_mlp_projector(config, in_dim):
    run = config['run']
    torch.manual_seed(run['seed'])
    projector = MLPProjector(in_dim, config['model']['projector_out_dim'])
    projector.to(device=run['device'], dtype=resolve_dtype(run['precision']))
    projector.eval()
    projector.requires_grad_(False)
    return projector

def select_image_features(outputs, config):
    features = outputs.hidden_states[config['model']['layer_index']]
    if not config['model']['keep_cls']:
        features = features[:, 1:, :]
    return features



if __name__ == '__main__':
    config = cfg.load_config()
    run = config['run']
    print('run:', run)

    model, processor = load_model_and_processor(config)
    projector = load_mlp_projector(config, model.config.hidden_size)

    dtype = resolve_dtype(run['precision'])
    paths = data.list_image_paths(config)
    print(f"found {len(paths)} images in {data.resolve_image_dir(config)}")

    for batch_paths in data.iter_batches(paths, run['batch_size']):
        images = data.load_images(batch_paths)
        pixel_values = data.preprocess(processor, images, run['device'], dtype)

        with torch.inference_mode():
            outputs = model(pixel_values, output_hidden_states=True)
            features = select_image_features(outputs, config)
            embeddings = projector(features)

        print(f"{[p.name for p in batch_paths]}")
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))


