# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            data.py
# CREATED:         10/15/2026
# REVISION:        1.0
# ======================================= #

from PIL import Image
from pathlib import Path


def load_paths(config):
    root = Path(config['data']['path'])             # stores the the path of the input data as a string
    paths = sorted(root.glob('*.jpg'))              # stores a list of paths for all images in the input data

    stride = config['data'].get('stride',1)         # extracts the step size from the profile config
    max_images = config['data'].get('max_images')   # extracts the max images from the profile config

    paths = paths[::stride]                         # Extended slice with a step
    if max_images is not None:
        paths = paths[:max_images]                  # Sets maximum number of images

    return paths


## Splits paths into batches of run.batch_size (last batch may be short)
def iter_batches(paths, batch_size):
    for start in range(0, len(paths), batch_size):
        yield paths[start:start + batch_size]

## Decode stage: file bytes -> RGB PIL images (drops alpha/grayscale so CLIP always sees 3 channels)
def load_images(paths):
    return [Image.open(path).convert('RGB') for path in paths]

## Preprocess stage: resize -> center crop -> normalize -> NCHW tensor on the target.
## nuScenes frames are 1600x900, so the shortest-edge resize gives 597x336 and the
## 336x336 center crop keeps only the middle ~56% of the horizontal field of view.
def preprocess(processor, images, target, dtype):
    batch = processor(images=images, return_tensors='pt')
    return target.to_device(batch['pixel_values'], dtype)
