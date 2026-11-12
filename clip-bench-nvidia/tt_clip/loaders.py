import os
from transformers import CLIPTokenizer, CLIPVisionModel

def download_model(model_name):

    return CLIPVisionModel.from_pretrained(model_name).eval()

