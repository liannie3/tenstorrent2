import numpy as np
import torch
import torchvision #forces ultralytics' NMS to use torchvision's GPU based NMS #TT CHANGE: NMS now runs on the CPU (see postprocess_stage)
import torch_xla #PyTorch/XLA for tt xla
                 #the "tt" torch.compile backend that compile_stages() uses, so it must be imported before compiling

from ultralytics.data.augment import LetterBox #ultralytics preprocess function to resize and pad #NOTE: runs on CPU
                                               #input images to a specified shape while preserving aspect ratio
                                               #additionally updates labels & bounding boxes respectively
from ultralytics.utils.nms import non_max_suppression #ultralyitics post process function that applies NMS on prediction results
                                                      #motivated to filter overlapping bounding boxes based on confidence and IoU (intersection over union) threshhold
from ultralytics.utils.ops import scale_boxes #ultralyitics utility function that rescales bounding boxes from one image to another #NOTE: runs on CPU

import config
#imports both the timed wrapper function and the timings dictionary from timing
from timing import timed, timings

compiled = {} #holds the torch compiled versions of each gpu substage ("normalize", "backbone", "neck", "head")
              #kept as separate graphs to allow for individual metrics
              #TODO: try and compare against a single graph

#######################################################################################################################################################################
#                        | | | |                                                                                                                                      #
#   Pre-Processing Stage V V V V                                                                                                                                      #
#######################################################################################################################################################################
#TODO: ultralytics' preprocess timer doesn't include reading images from disk,
#            so read before timer starts
#Passes a list of numpy tensors as im

def preprocess_stage(im: list[np.ndarray], device = config.DEVICE):
    store = timings["pre_processing"]
    orig_shapes = [x.shape for x in im]  #needed later by postprocess to rescale boxes

    #PRE-PROCESS STEP A: pre-transform input img before inference
    #NOTE: occurs before host to device (GPU) transfer so not GPU work
    with timed(store, "A_LetterBox", device, gpu_work = False):
        same_shapes = len({x.shape for x in im}) == 1 #determine if tensors are of same dimension/size
        letterbox = LetterBox( new_shape = (config.IMGSZ,config.IMGSZ), auto = same_shapes,  stride = 32 )
        resized_imgs = [letterbox(image = x) for x in im] #resize each image passed in the current batch

    #PRE-PROCESS STEP B: batch numpy stacking
    #NOTE: occurs before host to device (GPU) transfer so not GPU work
    with timed(store, "B_batch_stack", device, gpu_work = False):
        if len(resized_imgs) == 1:
            im = torch.from_numpy(resized_imgs[0]).unsqueeze(0)
        else:
            im = torch.from_numpy(np.stack(resized_imgs) )

    #PRE-PROCESS STEP C: host to memory transfer batch transfer to gpu
    with timed(store, "C_host_to_gpu", device):
        im = im.permute(0, 3, 1, 2).flip(1).contiguous().to(device)
        #NOTE: reorder BHWC -> BCHW and BGR -> RGB on the cpu before copying to card
        #BEST EXPLANATION: reorder moved to this step since the bh p150a's memory is organized in 32x32 tiles, thus last two dimensions of im's shape must be padded to multiples of 32
        #so copying BWHC boxes (with their final dim being 3) would need to be padded to [B,H,W, 32] which is 10x the data
        #instead if BCHW is passed to device, then HW wont require padding since they're already mutliples of stride = 32 from ultralyitcs letter box func

        #adds some time to step C for TT run, but more objective comparison
        #TODO: verify if we should be changing this between CUDA and TT run with TA

    #PRE-PROCESS STEP D: adjust and normalize tensor layout (GPU)
    with timed(store, "D_normalize_layout", device):
        im = compiled["normalize"](im) #moved previous operations into one function so can
                                       # now run as one compiled graph

    return im, orig_shapes #return img tensor after pre processing is applied to it, and orignal tensor for post processing

#consolidated the normalization substage's operations into a single function so it can be captured as its own graph during compilatione
def normalize_layout(im):
    #im = im.permute(0, 3, 1, 2) #transfer from torch standard BHWC (batch, height, width, channels)
                                #to ultralytics standard BCHW (batch, channels, height, width) tensor dimension representation
    #NOTE: performing on CPU before host to device transfer to prevent slow padding from 3 -> multiple of 32 on device which was heavily slowing down this normalization ste
    #TODO: confirm with saby if this is something to do for an objective comparison between CUDA and TT

    #if im.shape[1] == 3:
    #    im = im.flip(1) # BGR to RGB if there are three channels
    #TODO: ask saby if we should do this for an objective comparison between CUDA and TT

    im = im.to(config.DTYPE)
    im = im.div_(255) #normalize tensor pixels from [0.0 - 255.0] -> [0.0 - 1.0]
    return im

########################################################################################################################################################################
#                   | | | |                                                                                                                                            #
#   Inference Stage V V V V                                                                                                                                            #
########################################################################################################################################################################

#heleper function for layer walk during inference stage, equivalent to _predict_once
def run_layers(model, layers, x, y):
    for m in layers:
        if m.f != -1: #if current layer not taking input from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x) #run layer wiht updated feature map
        y.append(x if m.i in model.save else None) #only append layers that are indicated to be used later
                                                   #since model.save indicates layers that are yet to be used
    return x

#arg of model, feature map tensor im, and device that model is to be run on
#Based on _predict_once method from ultralytics/nn/tasks.py
# NOTE: i think just iterates through the different layers present underneath the inference model
#       if further granularity is requested by saby we can maybe pinpoint which layer is associated with
#       which submodel? Similarly to how we identified spatial pooling to be the 9th layer
def inference_stage(model, im, device = config.DEVICE): #TT CHANGE: default device was 'cuda:0'
    store = timings["inference"]
    y = [] #store feature map tensors from earlier layers for future use
    x = im
    #INFERENCE STEP A: BACKBONE extract features from input at multiple resolutions
    #C2F (CSP (Constraint Satisfaction Problem) Bottleneck with 2 convolutions, faster)
    with timed(store, "A_backbone", device):
        x = compiled["backbone"](x, y) #runs backbone as one compiled graph
        #NOTE: loops through each layer interior to yolo until reaching the backbone stage's boundary
        #NOTE: potential inference step: SPATIAL POOLING which is considered separate from the backbone occurs
        #                        at layer 9 inside the model might want to pull that one out of backbone

    #INFERENCE STEP B: NECK fuse features across scales so both large and small objects have context
    #                  occurs in layers 10-21 of model
    with timed(store, "B_neck", device):
        x = compiled["neck"](x, y) #runs neck as one compiled graph
        #NOTE: loops through each layer interior to yolov8 until reaching the backbone stage's boundary

    #INFERENCE STEP D: HEAD predict bounding boxes and class scores from fused features
    #                  occurs in layer 22
    with timed(store, "C_head", device):
        x = compiled["head"](x, y) #runs HEAD layer as one compiled graph
    return x #returns tuple of y (inferenced tensor), preds (dictionary of head layer's raw input)

########################################################################################################################################################################
#                         | | | |                                                                                                                                      #
#   Post-Processing Stage V V V V                                                                                                                                      #
########################################################################################################################################################################

#based on postprocess function from ultralytics/models/yolo/detect/predict.py
#NOTE: post-processing is deliberately left eager (not compiled), the same as ultralytics' own predict(compile=True),
#      which compiles the model only. How many boxes survive NMS depends on the image, and torch.compile handles
#      data-dependent output sizes poorly (graph breaks and recompiles), so compiling it would add noise, not speed
def postprocess_stage(preds, im, orig_shapes, conf_thres: float = 0.25, iou_thres: float = 0.7, device = config.DEVICE): #TT CHANGE: default device was 'cuda:0'
    store = timings["post_processing"]

    #NOTE: NMS not feasible to execute on bh p150a device since variation of how many boxes pass the confidence and IOU intervals
    #      leads to graph shapes changing between image which violates tt's fixed shape compiler
    #      need to copy results back to host in order for NMS and rescaling to be executed on host
    #NOTE: CUDA doesnt require this step since we can run NMS in eager mode (which TT software stack can't facilitate) and not violate any compiled graphs
    with timed(store, "A_device_to_host", device):
        preds = preds[0].cpu().float()

    #POST PROCESS STEP B: apply NMS to get single best prediction for each detected object, i.e. no redundant bounding boxes
    with timed(store, "B_NMS", device, gpu_work = False):
        filtered_preds = non_max_suppression(
            preds,
            conf_thres,
            iou_thres,
            classes = None,
            agnostic = False,
            multi_label = False,
            max_det = 300,
            nc = 0,  # number of classes (optional)
            end2end = False,
            rotated = False,
        )

    #POST PROCESS STEP C: Rescale each img's boxes from letterboxed space to original image space
    #Coordinate Rescaling using ultralytics/utils/ops.py
    with timed(store, "C_rescale_boxes", device, gpu_work = False):
        for pred, shape in  zip(filtered_preds, orig_shapes):
            if len(pred):
                pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], shape) #maps coordinates back from 640x640 space to the original image shape

    return filtered_preds #returns the list of result objects containing the post processed prediction

########################################################################################################################################################################
#                  | | | |                                                                                                                                             #
#   Compile Stages V V V V                                                                                                                                             #
########################################################################################################################################################################

#(NOTE: following attempt_compile function in ultralyitics/utils/torch_utils.py)
# called following the loading of model
# wraps each gpu substep in a torch.compile with settings consistent with ultralyitis standards (e.g. no mode or options and backend = "tt")
# defaults only i.e. no optimization_level, no runtime trace, no bfp8 weights, TT's optional performance settings stay off,
# small operations fused into generated kernels, reduces a lot of kernel launch overhead
# NOTE: Each stage is compiled on its own instead of the whole model at once so the A_backbone/B_neck/C_head timers
#       still have real boundaries to measure between (split the TT-XLA version at the same layers)
# NOTE: dynamo still records each graph, and the tt backend turns each captured graph into StableHLO,
#       compiles it with tt-mlir into TT-NN ops, and runs those on the card

# TODO: compare discretized graph setup vs one compiled gpu graph

def compile_stages(model):

    #slice each stages boundaries once here before work begins
    #avoids recompiling each model during every run (like slicing multiple times would cause)
    backbone = model.model[: config.BACKBONE_BOUNDARY+1]
    neck = model.model[config.BACKBONE_BOUNDARY+1: config.NECK_BOUNDARY+1]
    head = model.model[config.NECK_BOUNDARY+1:]

    #one separate function to represent each stage
    #separate  so each stage keeps its own cache of compiled graphs
    def backbone_fn(x, y): return run_layers(model, backbone, x, y)
    def neck_fn(x, y): return run_layers(model, neck, x, y)
    def head_fn(x, y): return run_layers(model, head, x, y)

    #dynamic=False: one fixed-shape graph per input shape, which is also how XLA-based backends like TT-XLA compile.
    #it stops torch.compile's default behaviour of swapping in a generic any-batch-size graph once it sees a second batch size.
    for name, fn in (("normalize", normalize_layout), ("backbone", backbone_fn), ("neck", neck_fn), ("head", head_fn)):
        compiled[name] = torch.compile(fn, backend = "tt", dynamic = False) #TT CHANGE: was mode = "default", backend = "inductor"

    # NOTE: torch.compile is lazy: nothing compiles until a stage's first call (the warmup in run_pipeline), so initial call
    #       takes seconds to minutes
    # TT NOTE: tt-mlir compiles every graph and JIT-builds its kernels on that first call; tt-metal caches the
    #          kernels on disk, so later runs start faster

