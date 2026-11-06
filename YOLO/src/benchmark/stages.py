import numpy as np
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
    
    return im, orig_shapes #return img tensor after pre processing is applied to it, and orignal tensor for post processing

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
        x = run_layers(model, model.model[: config.BACKBONE_BOUNDARY+1], x, y)
        #NOTE: loops through each layer interior to yolo until reaching the backbone stage's boundary
        #NOTE: potential inference step: SPATIAL POOLING which is considered separate from the backbone occurs
        #                        at layer 9 inside the model might want to pull that one out of backbone

    #INFERENCE STEP B: NECK fuse features across scales so both large and small objects have context
    #                  occurs in layers 10-21 of model
    with timed(store, "B_neck", device):
        x = run_layers(model, model.model[config.BACKBONE_BOUNDARY+1: config.NECK_BOUNDARY+1], x, y)
        #NOTE: loops through each layer interior to yolov8 until reaching the backbone stage's boundary
                
    #INFERENCE STEP D: HEAD predict bounding boxes and class scores from fused features
    #                  occurs in layer 22
    with timed(store, "C_head", device):
        x = run_layers(model, model.model[config.NECK_BOUNDARY+1:], x, y) #runs HEAD layer only
    return x #returns tuple of y (inferenced tensor), preds (dictionary of head layer's raw input)

########################################################################################################################################################################
#                         | | | |                                                                                                                                      #
#   Post-Processing Stage V V V V                                                                                                                                      #
########################################################################################################################################################################

#based on postprocess function from ultralytics/models/yolo/detect/predict.py
def postprocess_stage(preds, im, orig_shapes, conf_thres: float = 0.25, iou_thres: float = 0.7, device = 'cuda:0'):
    store = timings["post_processing"]
    
    #POST PROCESS STEP A: apply NMS to get single best prediction for each detected object, i.e. no redundant bounding boxes
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
    with timed(store, "B_rescale_boxes",device):
        for pred, shape in  zip(filtered_preds, orig_shapes):
            if len(pred):
                pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], shape) #maps coordinates back from 640x640 space to the original image shape

    return filtered_preds #returns the list of result objects containing the post processed prediction

