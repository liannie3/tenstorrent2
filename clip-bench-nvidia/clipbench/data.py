from PIL import Image
from pathlib import Path

def load_paths(config):
    root = Path(config['data']['path'])
    paths = sorted(root.glob('*.jpg'))

    stride = config['data'].get('stride',1)
    max_images = config['data'].get('max_images')

    paths = paths[::stride]
    if max_images is not None:
        paths = paths[:max_images]

    return paths

## Splits paths into batches of run.batch_size (last batch may be short)
def iter_batches(paths, batch_size):
    for start in range(0, len(paths), batch_size):
        yield paths[start:start + batch_size]

## Decode stage: file bytes -> RGB PIL images (drops alpha/grayscale so CLIP always sees 3 channels)
def load_images(paths):
    return [Image.open(path).convert('RGB') for path in paths]

## Preprocess stage: resize -> center crop -> normalize -> NCHW tensor on the run device.
## nuScenes frames are 1600x900, so the shortest-edge resize gives 597x336 and the
## 336x336 center crop keeps only the middle ~56% of the horizontal field of view.
def preprocess(processor, images, device, dtype):
    batch = processor(images=images, return_tensors='pt')
    return batch['pixel_values'].to(device=device, dtype=dtype)
