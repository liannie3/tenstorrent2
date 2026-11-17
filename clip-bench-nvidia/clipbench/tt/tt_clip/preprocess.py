from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

def preprocess_image(image, model_resolution):
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    # Pre-process image on host with torch
    transform_fn = Compose(
        [
            Resize(model_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(model_resolution),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    return transform_fn(image)