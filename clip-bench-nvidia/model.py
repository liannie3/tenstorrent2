import torch
import torch.nn as nn
from transformers import CLIPImageProcessorPil, CLIPVisionModel

import harness.config as cfg
import harness.data as data
import harness.stagetiming as timing

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


## Batch sizes to sweep: one sub-test each, every sub-test over the same image set
def resolve_batch_sizes(run):
    batch_sizes = run.get('batch_sizes')
    if batch_sizes is None:
        batch_sizes = [run['batch_size']]     # single-batch-size configs still work
    return list(batch_sizes)


## One sub-test: the whole image set at a single batch size
def run_subtest(config, model, processor, projector, paths, batch_size):
    run = config['run']
    device = run['device']
    dtype = resolve_dtype(run['precision'])
    max_batches = run.get('max_batches')

    images_seen = 0
    shapes = None
    for i, batch_paths in enumerate(data.iter_batches(paths, batch_size)):
        if max_batches is not None and i >= max_batches:
            break

        timing.mark(batch_size, i, 'iter_start', device)
        images = data.load_images(batch_paths)
        timing.mark(batch_size, i, 'decode', device)

        pixel_values = data.preprocess(processor, images, device, dtype)
        timing.mark(batch_size, i, 'preprocess', device)

        with torch.inference_mode():
            outputs = model(pixel_values, output_hidden_states=True)
            features = select_image_features(outputs, config)
            timing.mark(batch_size, i, 'encode', device)

            embeddings = projector(features)
            timing.mark(batch_size, i, 'project', device)

        images_seen += len(batch_paths)
        shapes = (tuple(pixel_values.shape), tuple(features.shape), tuple(embeddings.shape))

    return timing.summarize(batch_size, images_seen), shapes


if __name__ == '__main__':
    config = cfg.load_config()
    run = config['run']
    print('run:', run)

    model, processor = load_model_and_processor(config)
    projector = load_mlp_projector(config, model.config.hidden_size)

    paths = data.load_paths(config)
    print(f"found {len(paths)} images in {config['data']['path']}")

    batch_sizes = resolve_batch_sizes(run)
    summaries = []
    for batch_size in batch_sizes:
        print(f"\n=== sub-test: batch size {batch_size} over {len(paths)} images ===")
        summary, shapes = run_subtest(config, model, processor, projector, paths, batch_size)
        if summary is None:
            print('  no iterations ran')
            continue

        summaries.append(summary)
        pixel_values_shape, features_shape, embeddings_shape = shapes
        print('  pixel_values:', pixel_values_shape)
        print('  features    :', features_shape)
        print('  projected   :', embeddings_shape)
        print(f"  batches     : {summary['batches']} ({summary['images']} images)")
        print(f"  mean batch  : {summary['mean_batch_ms']:.2f} ms")
        print(f"  throughput  : {summary['images_per_s']:.2f} img/s")
        stages = '  '.join(f'{stage}={ms:.2f}' for stage, ms in summary['stage_mean_ms'].items())
        print(f"  stages (ms) : {stages}")

    results_dir = config['output']['results_dir']
    print('\n=== sweep ===')
    print(f"{'batch':>5}  {'img/s':>9}  {'ms/batch':>9}")
    for summary in summaries:
        print(f"{summary['batch_size']:>5}  {summary['images_per_s']:>9.2f}  {summary['mean_batch_ms']:>9.2f}")

    if summaries:
        print('trace  :', timing.dump(results_dir=results_dir))
        print('summary:', timing.dump_summary(summaries, results_dir=results_dir))

