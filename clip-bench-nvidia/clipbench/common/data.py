# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            data.py
# CREATED:         10/15/2026
# REVISION:        1.0
# ======================================= #

from PIL import Image
from pathlib import Path


## One image list per camera, keyed by view name in data.view_order. Takes the
## first `views` cameras, so views=1 is always CAM_FRONT and views=6 the whole rig.
def load_views(config, views):
    root = Path(config['data']['path'])             # stores the the path of the input data as a string
    order = config['data']['view_order']            # canonical camera order, so a view count is reproducible

    if views > len(order):
        raise ValueError(f'run.views asked for {views} views but data.view_order lists '
                         f'{len(order)}: {order}')

    stride = config['data'].get('stride',1)         # extracts the step size from the profile config
    max_images = config['data'].get('max_images')   # extracts the max images from the profile config

    views_paths = {}
    for name in order[:views]:
        paths = sorted((root / name).glob('*.jpg'))  # every frame this camera recorded
        paths = paths[::stride]                      # Extended slice with a step
        if max_images is not None:
            paths = paths[:max_images]               # Sets maximum number of images
        views_paths[name] = paths

    return views_paths


## Frames line up across cameras -- the nuScenes filename timestamps differ by a
## few ms per camera, but the i-th file in each view is the same sample.
def num_frames(views_paths):
    return min((len(paths) for paths in views_paths.values()), default=0)


## One iteration's worth of work: frame i seen from each camera, in view order.
## zip stops at the shortest view, so a frame is never half a rig.
def iter_frames(views_paths):
    return zip(*views_paths.values())

## Decode stage: file bytes -> RGB PIL images (drops alpha/grayscale so CLIP always sees 3 channels)
def load_images(paths):
    return [Image.open(path).convert('RGB') for path in paths]

## Preprocess stage: resize -> center crop -> normalize -> NCHW tensor on device.
## nuScenes frames are 1600x900, so the shortest-edge resize gives 597x336 and the
## 336x336 center crop keeps only the middle ~56% of the horizontal field of view.
def preprocess(processor, images, device, dtype):
    batch = processor(images=images, return_tensors='pt')
    return batch['pixel_values'].to(device=device, dtype=dtype)
