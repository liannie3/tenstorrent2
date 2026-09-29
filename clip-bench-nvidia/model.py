import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionModel
import harness.config as cfg

def load_model_and_processor():
    model = CLIPVisionModel.from_pretrained(cfg.load_config()['model']['name'])
    processor = CLIPImageProcessor.from_pretrained(cfg.load_config()['model']['name'], torch_dtype=)
    return model, processor

def load_mlp_projector()

