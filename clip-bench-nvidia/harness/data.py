from pathlib import Path

from PIL import Image

## Extensions picked up from the camera directory (nuScenes ships .jpg)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

## nuScenes keyframes live at <root>/samples/<CAM_*>; sweeps/ holds the unannotated 12Hz frames
def resolve_image_dir(config):
    data = config['data']
    root = Path(data['nuscenes_root'])
    if not root.is_dir():
        raise FileNotFoundError(f"data.nuscenes_root {root} does not exist")

    samples = root / 'samples'
    if not samples.is_dir():
        raise FileNotFoundError(
            f"{root} has no samples/ directory - point data.nuscenes_root at the "
            f"extracted v1.0-mini root, which contains samples/, sweeps/ and maps/"
        )

    image_dir = samples / data['camera']
    if not image_dir.is_dir():
        available = sorted(p.name for p in samples.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"camera {data['camera']!r} not found in {samples}, available: {available}"
        )
    return image_dir

## Collects keyframe paths, then applies data.stride and data.max_images
def list_image_paths(config):
    data = config['data']
    image_dir = resolve_image_dir(config)

    ## Sorted = reproducible batches; nuScenes filenames end in a timestamp, so each
    ## log's frames come out in capture order
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images in {image_dir}, expected one of {sorted(IMAGE_EXTS)}")

    stride = data.get('stride') or 1
    if stride < 1:
        raise ValueError(f"data.stride must be >= 1, got {stride}")
    paths = paths[::stride]

    max_images = data.get('max_images')
    if max_images is not None:
        if max_images < 1:
            raise ValueError(f"data.max_images must be >= 1 or null, got {max_images}")
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
