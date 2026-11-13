import ttnn
import torch 
import math

from device import get_device
from vision import VisionTransformer
from attention import Transformer


class TTClipVisionModel:
    def __init__(self, state_dict, num_layers):
        self.visual = VisionTransformer(state_dict, num_vision_layers=num_layers)


    def forward(self, image):
        x = self.visual.forward(image)                 # [batch, 577, 1024]
        batch, seq, hidden = x.shape
        return ttnn.slice(x, [0, 1, 0], [batch, seq, hidden])  # drop CLS -> [batch, 576, 1024]