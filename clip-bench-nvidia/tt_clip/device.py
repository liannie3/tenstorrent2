import ttnn
import torch
from transformers import CLIPVisionModel

device = None   # module-level handle; open_ttnn() fills it in

def open_ttnn(device_id=0, l1_small_size=8192, **kwargs):
    """Initialize TT-NN device with specified L1 cache size.

    Extra keyword arguments go straight to ttnn.open_device -- clipbench's
    run_tt.py passes trace_region_size so it can replay Metal traces.
    """
    global device
    device = ttnn.open_device(device_id=device_id, l1_small_size=l1_small_size, **kwargs)

def close_ttnn():
    """Clean up and close the TT-NN device."""
    global device
    if device is not None:
        ttnn.close_device(device)
        device = None

def get_device():
    """Get the current TT-NN device handle."""
    return device