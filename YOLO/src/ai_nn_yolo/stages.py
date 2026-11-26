import numpy as np
import torch
import torchvision  # noqa: F401  #forces ultralytics' NMS to use torchvision's NMS (CPU here: the card hands the boxes back to the host)

import ttnn
from ultralytics.data.augment import LetterBox #ultralytics preprocess function to resize and pad #NOTE: runs on CPU
from ultralytics.utils.nms import non_max_suppression #ultralyitics post process function that applies NMS on prediction results
from ultralytics.utils.ops import scale_boxes #rescales bounding boxes from the letterboxed image to the original #NOTE: runs on CPU

import config
from timing import timed, timings

#######################################################################################################################################################################
#                        | | | |                                                                                                                                      #
#   Pre-Processing Stage V V V V                                                                                                                                      #
#######################################################################################################################################################################
#NOTE: ultralytics' preprocess timer doesn't include reading images from disk, so images are read before the timer starts
#Passes a list of numpy images (HWC, BGR, uint8) as im

def preprocess_stage(im: list[np.ndarray], ctx, model, input_hw=None):
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
    #NOTE: BGR->RGB and BHWC->BCHW happen here in numpy, exactly where ultralytics' own preprocess does them.
    #      uint8 -> bf16 also happens here (0-255 are exact in bf16)
    with timed(store, "B_batch_stack", device, device_work=False):
        batch = np.stack(resized_imgs) #[N, H, W, 3] uint8 BGR
        batch = np.ascontiguousarray(batch[..., ::-1].transpose(0, 3, 1, 2)) #BGR to RGB, BHWC to BCHW
        batch = torch.from_numpy(batch).to(torch.bfloat16)

    #PRE-PROCESS STEP C: host to device transfer (PCIe) into DRAM, row-major
    with timed(store, "C_host_to_device", device):
        x = ttnn.from_torch(batch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

    #PRE-PROCESS STEP D: adjust tensor layout (on the card)
    #NOTE: pad 3 -> 16 channels, NCHW -> NHWC, flatten to [1, 1, N*H*W, 16]: the first lines of the yolov8x demo's forward.
    #      the [0 - 255] -> [0.0 - 1.0] scale is folded into the stem conv's weights (conv(x/255) == (W/255) conv x), so
    #      there is no on-card multiply (a bf16 1/255 rounds to 1/256, a 0.4% input scale error)
    with timed(store, "D_normalize_layout", device):
        x = model.prepare_input(x)

    return x, orig_shapes #pixel values are still 0-255 here

########################################################################################################################################################################
#                   | | | |                                                                                                                                            #
#   Inference Stage V V V V                                                                                                                                            #
########################################################################################################################################################################

#arg of TT model, flattened NHWC input from preprocess_stage, and the device it runs on
#NOTE: the yolov8x demo runs the whole network in one forward; the port splits it at the same layer boundaries the CUDA
#      version uses (backbone 0-9, neck 10-21, head 22) so each part can be timed
def inference_stage(model, im, device):
    store = timings["inference"]
    #INFERENCE STEP A: BACKBONE extract features from input at multiple resolutions (layers 0-9, incl. SPPF at 9)
    with timed(store, "A_backbone", device):
        x = model.backbone(im)

    #INFERENCE STEP B: NECK fuse features across scales so both large and small objects have context (layers 10-21)
    with timed(store, "B_neck", device):
        x = model.neck(x)

    #INFERENCE STEP C: HEAD predict bounding boxes and class scores from fused features (layer 22, incl. the box decode)
    with timed(store, "C_head", device):
        x = model.head(x)
    return x #[decoded predictions [N, 84, anchors], per-level head maps], still on the card

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
        prediction = ttnn.to_torch(preds[0]).float() #[N, 4 + nc, anchors], the format non_max_suppression expects
        for t in [preds[0], *preds[1]]: #free this batch's head outputs on the card
            ttnn.deallocate(t)

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
