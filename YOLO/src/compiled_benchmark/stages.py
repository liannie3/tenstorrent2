import numpy as np
import sys #used by compile_stages() to stop the run if torch.compile doesn't work on machine
import torch
import torchvision #forces ultralytics' NMS to use torchvision's GPU based NMS

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

def preprocess_stage(im: list[np.ndarray], device = 'cuda:0', fp16 = False):
    store = timings["pre_processing"]
    orig_shapes = [x.shape for x in im]  #needed later by postprocess to rescale boxes
    
    #PRE-PROCESS STEP A: pre-transform input img before inference
    #NOTE: occurs before host to device (GPU) transfer so not GPU work
    with timed(store, "A_LetterBox", device, gpu_work = False):
        same_shapes = len({x.shape for x in im}) == 1 #determine if tensors are of same dimension/size
        letterbox = LetterBox( new_shape = (config.IMGSZ,config.IMGSZ), auto = same_shapes,  stride = 32 ) #TODO: might need to load stride from model itself
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
        im = im.to(device) #tranfer tensors to gpu

    #PRE-PROCESS STEP D: adjust and normalize tensor layout (GPU)
    with timed(store, "D_normalize_layout", device):
        im = compiled["normalize"](im, fp16) #run normalization stage as one compiled graph
    
    return im, orig_shapes #return img tensor after pre processing is applied to it, and orignal tensor for post processing

#consolidated the normalization substage's operations into a single function so it can be captured as its own graph during compilation
def normalize_layout(im, fp16):
    im = im.permute(0, 3, 1, 2) #transfer from torch standard BHWC (batch, height, width, channels)
                                #to ultralytics standard BCHW (batch, channels, height, width) tensor dimension representation
    if im.shape[1] == 3:
        im = im.flip(1) # BGR to RGB if there are three channels
    im = im.contiguous()  #allocate a new contiguous (i.e. uninterrupted) block of memory and copy data into it
    if fp16:
        im = im.half() #if fp16 (16 bit float), convert image pixels tensor from unint8 to fp16
    else:
        im = im.float() #if not fp16 convert image pixels tensor from uint8 to fp32 (32 bit float)
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
def inference_stage(model, im, device = 'cuda:0'):
    store = timings["inference"]
    y = [] #store feature map tensors from earlier layers for future use
    x = im
    #INFERENCE STEP A: BACKBONE extract features from input at multiple resolutions
    #C2F (CSP (Constraint Satisfaction Problem) Bottleneck with 2 convolutions, faster)
    with timed(store, "A_backbone", device):
        x = compiled["backbone"](x, y) #run backbone layer as one compiled graph
        
        #NOTE: loops through each layer interior to yolo until reaching the backbone stage's boundary
        #NOTE: potential inference step: SPATIAL POOLING which is considered separate from the backbone occurs
        #                        at layer 9 inside the model might want to pull that one out of backbone

    #INFERENCE STEP B: NECK fuse features across scales so both large and small objects have context
    #                  occurs in layers 10-21 of model
    with timed(store, "B_neck", device):
        x = compiled["neck"](x, y) #run neck layer as one compiled graph
        #NOTE: loops through each layer interior to yolov8 until reaching the backbone stage's boundary
                
    #INFERENCE STEP D: HEAD predict bounding boxes and class scores from fused features
    #                  occurs in layer 22
    with timed(store, "C_head", device):
        x = compiled["head"](x, y) #runs head layer as one compiled graph
    return x #returns tuple of y (inferenced tensor), preds (dictionary of head layer's raw input)

########################################################################################################################################################################
#                         | | | |                                                                                                                                      #
#   Post-Processing Stage V V V V                                                                                                                                      #
########################################################################################################################################################################

#based on postprocess function from ultralytics/models/yolo/detect/predict.py
#COMPILE NOTE: post-processing is deliberately left eager (not compiled), the same as ultralytics' own predict(compile=True),
#              which compiles the model only. How many boxes survive NMS depends on the image, and torch.compile handles
#              data-dependent output sizes poorly (graph breaks and recompiles), so compiling it would add noise, not speed
def postprocess_stage(preds, im, orig_shapes, conf_thres: float = 0.25, iou_thres: float = 0.7, device = 'cuda:0'):
    store = timings["post_processing"]
    
    #POST PROCESS STEP A: apply NMS to get single best prediction for each detected object, i.e. no redundant bounding boxes
    #NOTE: NOT CAPTURED AS A COMPILED GRAPH since nms function contains code that breaks graph format
    #      Additionally shape of data would change from batch to batch requiring recompilation
    #TODO: TT implementation requires NMS be done on the CPU since varying input sizes breaks graph standards
    with timed(store, "A_NMS", device):
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
    
    #POST PROCESS STEP B: Rescale each img's boxes from letterboxed space to original image space  
    #Coordinate Rescaling using ultralytics/utils/ops.py
    #NOTE: NOT CAPTURED AS A COMPILED GRAPH since rescaling  contains code that breaks graph format
    #      Additionally shape of data would change from batch to batch requiring recompilation
    with timed(store, "B_rescale_boxes",device):
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
# wraps each gpu substep in a torch.compile with settings consistent with ultralyitis standards (e.g. mode = default and backend = "inductor")
# meaning use of stock torch.compile: no autotuning ("max-autotune") and no CUDA graphs ("reduce-overhead").
# small operations fused into generated kernels, reduces a lot of kernel launch overhead
# NOTE: Each stage is compiled on its own instead of the whole model at once so the A_backbone/B_neck/C_head timers
#       still have real boundaries to measure between (split the TT-XLA version at the same layers)
# TODO: compare discretized graph setup vs one compiled gpu graph
def compile_stages(model):

    #stop rather than silently time eager code under the "compiled" label 
    if torch.device(config.DEVICE).type == "cuda":
        from torch.utils._triton import has_triton #torch.compile needs Triton to generate CUDA kernels
        if not has_triton():
            sys.exit("ERROR: NO WORKING TRITON INSTALL FOUND, torch.compile NEEDS IT TO GENERATE CUDA KERNELS")

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
    #stops torch.compile's default behaviour of swapping in a generic any-batch-size graph once it sees a second batch size
    #NOTE: (hardware_substages runs batch 25 and then batch 1). Use the same setting on the TT-XLA side
    for name, fn in (("normalize", normalize_layout), ("backbone", backbone_fn), ("neck", neck_fn), ("head", head_fn)):
        compiled[name] = torch.compile(fn, mode = "default", backend = "inductor", dynamic = False)

    # NOTE: torch.compile by default is lazy, meaning  nothing compiles until a stage's first call (the warmup in run_pipeline), 
            #first call so that call takes seconds to minutes. Inductor caches what it builds on disk, so later runs start faster
 

