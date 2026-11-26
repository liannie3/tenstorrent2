# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Ported from tt-metal v0.64.0 models/demos/yolov8x/tt/ttnn_yolov8x_utils.py (yolov8x -> yolov8n)
# NOTE: changes from the original are marked with "#NOTE: yolov8n port:"
#   - the conv list is built from the state dict instead of a hard-coded yolov8x list (yolov8n has fewer bottlenecks)
#   - the /255 input scale is folded into the stem conv weights (the harness hands the card 0-255 pixels)
#   - determine_num_cores / get_core_grid_from_num_cores are copied in from models/experimental/yolo_common/yolo_utils.py
#     (Apache-2.0, © 2025 Tenstorrent USA, Inc.) so this folder runs outside the tt-metal source tree

import math

import torch

import ttnn


def fold_batch_norm2d_into_conv2d_split(device, state_dict, path, eps=1e-03, bfloat8=False, mesh_mapper=None):
    bn_weight = state_dict[path + f".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = state_dict[path + f".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = state_dict[path + f".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = state_dict[path + f".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = state_dict[path + f".conv.weight"]
    weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bias = bias.reshape(1, 1, 1, -1)

    chunk_size = bias.shape[-1] // 2

    if bfloat8:
        return (
            ttnn.from_torch(weight[:chunk_size, :, :, :], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(bias[:, :, :, :chunk_size], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(weight[chunk_size:, :, :, :], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(bias[:, :, :, chunk_size:], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
        )

    return (
        ttnn.from_torch(weight[:chunk_size, :, :, :], dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        ttnn.from_torch(bias[:, :, :, :chunk_size], dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        ttnn.from_torch(weight[chunk_size:, :, :, :], dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        ttnn.from_torch(bias[:, :, :, chunk_size:], dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
    )


def fold_batch_norm2d_into_conv2d(device, state_dict, path, eps=1e-03, bfloat8=False, mesh_mapper=None, in_scale=None):
    bn_weight = state_dict[path + f".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = state_dict[path + f".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = state_dict[path + f".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = state_dict[path + f".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = state_dict[path + f".conv.weight"]
    weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bias = bias.reshape(1, 1, 1, -1)

    #NOTE: yolov8n port: conv(x / 255, W) == conv(x, W / 255), so the stem takes 0-255 pixels with no on-card multiply
    if in_scale is not None:
        weight = weight * in_scale

    if bfloat8:
        return (
            ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper),
        )

    return (
        ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        ttnn.from_torch(bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
    )


def make_anchors(device, feats, strides, grid_cell_offset=0.5, mesh_mapper=None):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i]
        sx = torch.arange(end=w) + grid_cell_offset
        sy = torch.arange(end=h) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride))

    a = torch.cat(anchor_points).transpose(0, 1).unsqueeze(0)
    b = torch.cat(stride_tensor).transpose(0, 1)

    return (
        ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper),
        ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper),
    )


def ttnn_decode_bboxes(device, distance, anchor_points, xywh=True, dim=1):
    distance = ttnn.to_layout(distance, ttnn.ROW_MAJOR_LAYOUT)
    # lt, rb = ttnn.split(distance, 2, 1, memory_config=ttnn.L1_MEMORY_CONFIG)  # if done in tile : tt-metal issue #17017

    lt = distance[:, : distance.shape[1] // 2, :]
    rb = distance[:, distance.shape[1] // 2 :, :]

    lt = ttnn.to_layout(lt, ttnn.TILE_LAYOUT)
    rb = ttnn.to_layout(rb, ttnn.TILE_LAYOUT)

    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = x1y1 + x2y2
        c_xy = ttnn.div(c_xy, 2)
        wh = x2y2 - x1y1
        return ttnn.concat([c_xy, wh], 1)


def preprocess_parameters(state_dict, path, bias=True, bfloat8=True, mesh_mapper=None):
    if bias:
        conv_weight = state_dict[f"{path}.2.weight"]
        conv_bias = state_dict[f"{path}.2.bias"]

        if bfloat8:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
            conv_bias = ttnn.reshape(
                ttnn.from_torch(conv_bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper), (1, 1, 1, -1)
            )
        else:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
            conv_bias = ttnn.reshape(
                ttnn.from_torch(conv_bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper), (1, 1, 1, -1)
            )

        return (conv_weight, conv_bias)

    else:
        conv_weight = state_dict[f"{path}.conv.weight"]

        if bfloat8:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        else:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

        return (conv_weight, None)


def custom_preprocessor(device, state_dict, inp_h=640, inp_w=640, mesh_mapper=None, bfloat8=True):
    #NOTE: yolov8n port: the original listed every yolov8x conv by hand (model.2.m.0 ... model.2.m.2, model.4.m.0 ... m.5, ...).
    #      yolov8n has 1-2 bottlenecks per C2f instead of 3-6, so the list is built from the state dict: every conv that has
    #      a BatchNorm after it (all of them except the last 1x1 of each Detect branch and the DFL conv)
    state_dict = {k: v.float() for k, v in state_dict.items()}  #yolov8n.pt ships fp16 weights
    pairs = [k[: -len(".bn.weight")] for k in state_dict if k.endswith(".bn.weight")]

    c2f_paths = [
        "model.2.cv1",
        "model.4.cv1",
        "model.6.cv1",
        "model.8.cv1",
        "model.12.cv1",
        "model.15.cv1",
        "model.18.cv1",
        "model.21.cv1",
    ]

    parameters = {}

    for path in pairs:
        parameters[path] = fold_batch_norm2d_into_conv2d(
            device,
            state_dict,
            path=path,
            bfloat8=bfloat8,
            mesh_mapper=mesh_mapper,
            in_scale=1.0 / 255.0 if path == "model.0" else None,  #NOTE: yolov8n port: /255 folded into the stem
        )

        if path in c2f_paths:
            parameters_modified = fold_batch_norm2d_into_conv2d_split(
                device, state_dict, path=path, bfloat8=bfloat8, mesh_mapper=mesh_mapper
            )
            parameters[path + "_a"] = parameters_modified[:2]
            parameters[path + "_b"] = parameters_modified[2:]

    for i in range(3):
        parameters[f"model.22.cv2.{i}"] = preprocess_parameters(
            state_dict, f"model.22.cv2.{i}", bfloat8=bfloat8, mesh_mapper=mesh_mapper
        )
        parameters[f"model.22.cv3.{i}"] = preprocess_parameters(
            state_dict, f"model.22.cv3.{i}", bfloat8=bfloat8, mesh_mapper=mesh_mapper
        )

    parameters["model.22.dfl"] = preprocess_parameters(
        state_dict, "model.22.dfl", bias=False, bfloat8=bfloat8, mesh_mapper=mesh_mapper
    )

    strides = [8, 16, 32]

    feats = [(inp_h // 8, inp_w // 8), (inp_h // 16, inp_w // 16), (inp_h // 32, inp_w // 32)]  # value depend on res

    anchors, strides = make_anchors(
        device, feats, strides, mesh_mapper=mesh_mapper
    )  # Optimization: Processing make anchors outside model run

    parameters["anchors"] = anchors
    parameters["strides"] = strides

    return parameters


#NOTE: yolov8n port: copied from models/experimental/yolo_common/yolo_utils.py (© 2025 Tenstorrent USA, Inc., Apache-2.0)
def determine_num_cores(nhw: int, width: int, max_cores=64) -> int:
    gcd_nhw_width = math.gcd(nhw, width)
    cores = nhw // gcd_nhw_width
    if cores > max_cores:
        for divisor in range(max_cores, 0, -1):
            if nhw % divisor == 0 and (nhw // divisor) % width == 0:
                cores = divisor
                break
    return cores


#NOTE: yolov8n port: copied from models/experimental/yolo_common/yolo_utils.py (© 2025 Tenstorrent USA, Inc., Apache-2.0)
def get_core_grid_from_num_cores(num_cores: int, grid_rows: int = 8, grid_cols: int = 8):
    rows = num_cores // grid_cols
    assert rows <= grid_rows, "Not enough cores for specified core grid"
    ranges = []
    if rows != 0:
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_rows - 1, rows - 1),
            )
        )
    remainder = num_cores % grid_rows
    if remainder != 0:
        assert rows + 1 <= grid_rows, "Not enough cores for specified core grid"
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, rows),
                ttnn.CoreCoord(remainder - 1, rows),
            )
        )
    return ttnn.CoreRangeSet({*ranges})
