import numpy as np
import torch
import torchvision  # noqa: F401  #forces ultralytics' NMS to use torchvision's NMS (CPU here: the card hands the boxes back to the host)

import ttnn
from ultralytics.data.augment import LetterBox #ultralytics preprocess function to resize and pad #NOTE: runs on CPU
from ultralytics.utils.nms import non_max_suppression #ultralyitics post process function that applies NMS on prediction results
from ultralytics.utils.ops import scale_boxes #rescales bounding boxes from the letterboxed image to the original #NOTE: runs on CPU

import config
from timing import timed, timings
from tt_yolo import IMG_CHANNELS_PADDED, Feat

#######################################################################################################################################################################
#                        | | | |                                                                                                                                      #
#   Pre-Processing Stage V V V V                                                                                                                                      #
#######################################################################################################################################################################
#NOTE: ultralytics' preprocess timer doesn't include reading images from disk, so images are read before the timer starts
#Passes a list of numpy images (HWC, BGR, uint8) as im

def preprocess_stage(im: list[np.ndarray], ctx, input_hw=None):
    store = timings["pre_processing"]
    device = ctx.device
    input_hw = tuple(input_hw or config.INPUT_HW)
    orig_shapes = [x.shape for x in im]  #needed later by postprocess to rescale boxes

    #PRE-PROCESS STEP A: pre-transform input img before inference
    #NOTE: auto=False pads every frame to exactly input_hw, because the TT kernels are compiled for one static shape
    with timed(store, "A_LetterBox", device, device_work=False):
        letterbox = LetterBox(new_shape=input_hw, auto=False, stride=32)
        resized_imgs = [letterbox(image=x) for x in im] #resize each image passed in the current batch

    #PRE-PROCESS STEP B: batch stacking on the host
    #NOTE: BGR->RGB and BHWC->BCHW happen here in numpy, exactly where ultralytics' own preprocess does them. The CUDA version
    #      did them on the GPU; on TT the NCHW upload + on-card pad/permute below is the input path tt-metal's YOLO ports use.
    #      uint8 -> bf16 also happens here (0-255 are exact in bf16); TT has no well-trodden uint8 path into conv inputs yet.
    with timed(store, "B_batch_stack", device, device_work=False):
        batch = np.stack(resized_imgs) #[N, H, W, 3] uint8 BGR
        batch = np.ascontiguousarray(batch[..., ::-1].transpose(0, 3, 1, 2)) #BGR to RGB, BHWC to BCHW
        batch = torch.from_numpy(batch).to(torch.bfloat16)

    #PRE-PROCESS STEP C: host to device transfer (PCIe) into DRAM, row-major
    with timed(store, "C_host_to_device", device):
        x = ttnn.from_torch(batch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

    #PRE-PROCESS STEP D: adjust and normalize tensor layout (on the card)
    #NOTE: the [0 - 255] -> [0.0 - 1.0] scale is folded into the stem conv's weights (conv(x/255) == (W/255) conv x): same
    #      math, zero runtime cost. An on-card `x * (1/255)` can round the scalar to bf16 (= 1/256), a 0.4% input scale
    #      error that PCC cannot detect.
    with timed(store, "D_normalize_layout", device):
        n, c, h, w = batch.shape
        #pad channels 3 -> 16 (conv input alignment), then NCHW -> NHWC: TT convs are channels-last
        x = ttnn.pad(x, ((0, 0), (0, IMG_CHANNELS_PADDED - c), (0, 0), (0, 0)), value=0.0,
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.permute(x, (0, 2, 3, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.reshape(x, (1, 1, n * h * w, IMG_CHANNELS_PADDED)) #flattened NHWC, the form ttnn.conv2d works in
        x = ctx.tile(x) #tilize (convs take tiled inputs) and place in the activation buffer

    return Feat(x, n, h, w, IMG_CHANNELS_PADDED), orig_shapes #pixel values are still 0-255 here

########################################################################################################################################################################
#                   | | | |                                                                                                                                            #
#   Inference Stage V V V V                                                                                                                                            #
########################################################################################################################################################################

#helper function for layer walk during inference stage, equivalent to _predict_once (unchanged from the CUDA version: the TT
#layers carry the same i / f attributes and the TT model carries the same save list)
def run_layers(model, layers, x, y, capture=None):
    for m in layers:
        if m.f != -1: #if current layer not taking input from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x) #run layer with updated feature map
        y.append(x if m.i in model.save else None) #only append layers that are indicated to be used later
        if capture is not None: #verification keeps every layer's output
            capture[m.i] = x
    return x

#arg of TT model, feature map from preprocess_stage, and the device it runs on
#Based on _predict_once method from ultralytics/nn/tasks.py
def inference_stage(model, im, device, capture=None):
    store = timings["inference"]
    y = [] #store feature maps from earlier layers for future use
    x = im
    #INFERENCE STEP A: BACKBONE extract features from input at multiple resolutions (layers 0-9, incl. SPPF at 9)
    with timed(store, "A_backbone", device):
        x = run_layers(model, model.model[: config.BACKBONE_BOUNDARY + 1], x, y, capture)

    #INFERENCE STEP B: NECK fuse features across scales so both large and small objects have context (layers 10-21)
    with timed(store, "B_neck", device):
        x = run_layers(model, model.model[config.BACKBONE_BOUNDARY + 1 : config.NECK_BOUNDARY + 1], x, y, capture)

    #INFERENCE STEP C: HEAD predict bounding boxes and class scores from fused features (layer 22, incl. the box decode)
    with timed(store, "C_head", device):
        x = run_layers(model, model.model[config.NECK_BOUNDARY + 1 :], x, y, capture)
    return x #HeadOutput: decoded predictions still on the card

########################################################################################################################################################################
#                         | | | |                                                                                                                                      #
#   Post-Processing Stage V V V V                                                                                                                                      #
########################################################################################################################################################################

#based on postprocess function from ultralytics/models/yolo/detect/predict.py
def postprocess_stage(preds, orig_shapes, ctx, input_hw=None, conf_thres: float = 0.25, iou_thres: float = 0.7):
    store = timings["post_processing"]
    device = ctx.device
    input_hw = tuple(input_hw or config.INPUT_HW)

    #POST PROCESS STEP A: device to host transfer of the head output (blocks until the card is done)
    #mirror of C_host_to_device; on CUDA the predictions never left the GPU before NMS
    with timed(store, "A_device_to_host", device):
        host = preds.to_host()
        if preds.decoded_on_device:
            prediction = preds.assemble(host) #[N, 4 + nc, anchors], a view + one small cat

    #POST PROCESS STEP A2 (only with --decode host): DFL softmax + anchor decode + sigmoid on the CPU in fp32
    if not preds.decoded_on_device:
        with timed(store, "A2_decode_host", device, device_work=False):
            prediction = preds.assemble(host)

    #POST PROCESS STEP B: apply NMS to get single best prediction for each detected object, i.e. no redundant bounding boxes
    with timed(store, "B_NMS", device, device_work=False):
        filtered_preds = non_max_suppression(
            prediction,
            conf_thres,
            iou_thres,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=300,
            nc=0,  # number of classes (optional)
            end2end=False,
            rotated=False,
        )

    #POST PROCESS STEP C: Rescale each img's boxes from letterboxed space to original image space
    with timed(store, "C_rescale_boxes", device, device_work=False):
        for pred, shape in zip(filtered_preds, orig_shapes):
            if len(pred):
                pred[:, :4] = scale_boxes(input_hw, pred[:, :4], shape) #maps coordinates back from input_hw to the original image

    return filtered_preds #list of [num_dets, 6] tensors (x1, y1, x2, y2, conf, cls) per image
