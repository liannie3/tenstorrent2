from loguru import logger
import ttn
import torch

def convert_model_to_ttnn(state_dict):
    """
    Convert a PyTorch model's state dictionary to TT-NN format.

    Args:
        state_dict: PyTorch model state dictionary containing weights and biases

    Returns:
        dict: State dictionary with tensors converted to TT-NN format
    """
    ttnn_state_dict = {}
    logger.info(f"Converting model to TT-NN format")

    # Convert each tensor in the state dictionary to TT-NN format
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert PyTorch tensors to TT-NN tensors
            ttnn_state_dict[key] = ttnn.from_torch(value, layout=ttnn.TILE_LAYOUT, device=get_device())
            ttnn_state_dict[key] = ttnn.typecast(ttnn_state_dict[key], dtype=ttnn.bfloat16)
        elif isinstance(value, torch.Size):
            # Convert PyTorch Size objects to TT-NN Size objects
            ttnn_state_dict[key] = ttnn.Size(value)

    return ttnn_state_dict