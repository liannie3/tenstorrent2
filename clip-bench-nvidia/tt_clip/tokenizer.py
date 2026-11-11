import os
from transformers import CLIPTokenizer, CLIPModel

def download_model(model_name):
    clip_model_location = model_name # By default, download from Hugging Face

    # If TTNN_TUTORIALS_MODELS_CLIP_PATH is set, use it as the cache directory to avoid requests to Hugging Face
    cache_dir = os.getenv("TTNN_TUTORIALS_MODELS_CLIP_PATH")
    if cache_dir is not None:
        clip_model_location = cache_dir

    # Load model weights (download if cache_dir was not set)
    model = CLIPModel.from_pretrained(clip_model_location)

    return model


def download_tokenizer(tokenizer_name):
    clip_tokenizer_location = tokenizer_name # By default, download from Hugging Face

    # If TTNN_TUTORIALS_MODELS_CLIP_PATH is set, use it as the cache directory to avoid requests to Hugging Face
    cache_dir = os.getenv("TTNN_TUTORIALS_MODELS_CLIP_PATH")
    if cache_dir is not None:
        clip_tokenizer_location = cache_dir

    tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_location)

    return tokenizer