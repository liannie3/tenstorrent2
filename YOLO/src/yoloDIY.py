import cv2 #open source computer vision libray utilized by ultralytics
import torch
import time
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
from ultralyitics import YOLO
from ultralytics.nn.tasks import attempt_load_weights #ultralyitics neural network function


#NOTE: runs on CPU
from ultralytics.data.augment import LetterBox #ultralytics preprocess function to resize and pad
                                               #input images to a specified shape while preserving aspect ratio
                                               #additionally updates labels & bounding boxes respectively

from ultralytics.utils.nms import non_max_suppression #ultralyitics post process function that applies NMS on prediction results
                                                      #motivated to filter overlapping bounding boxes based on confidence and IoU (intersection over union) threshhold 

#NOTE: runs on CPU                                   
from ultralytics.utils.ops import scale_boxes #ultralyitics utility function that rescales bounding boxes from one image to another

#to store timing for each layer of each stage
STAGES = ("pre_processing", "inference", "post_processing")
timings = {s: defaultdict(list) for s in STAGES} #create dictionary to hold timing data

#MODEL SPECIFIC GLOBALS 
#TODO: currently hardcoded but a more robust option could be developed using the command line 
WEIGHTS = "yolov8n.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ = 640 #640 px
BATCH_SIZE = 1 #set to 1 by default in ultralytics predict func
CONF_THRES = 0.25 #confindence threshold for NMS 
IOU_THRESH = 0.75 #intersection over union threshold for NMS
FP16 = FALSE
#boundary layer indicies from ultralytics/cfg/models/v8/yoloe-v8.yaml
BACKBONE_BOUNDARY = 9 #indicates backbone layers are 0-9 
NECK_BOUNDARY = 21 #indicates neck layers are 10-21.

def _sync(device): #only waits  for transfer to gpu if we're using gpu in the first place
    if torch.device(device).type == "cuda": 
        torch.cuda.synchronize()
    #NOTE: needs to be adjusted for TT

#timer helper
#sync's before and after to isolate current step's GPU work (i.e. no spill over from previous step)
#runs setup code before the block and runs exit code following the block
@contextmanager
def timed(store, name, device):
    _sync(device) #flush previous GPU work
    t0 = time.perf_counter()
    yield  #exit code below 
    _sync(device) #include this step's GPU work
    store[name].append((time.perf_counter() - t0 ) * 1000) #convert to ms timing and store in dictionary at associated key

   
#                   Passes a list of numpy tensors as im
#TODO: ultralytics' preprocess timer doesn't include reading images from disk, 
#            so read before timer starts
def preprocess_stage(im: list[np.ndarray], device = 'cuda:0', fp16 = False):
    store = timings["pre_processing"]
    orig_shapes = [x.shape for x in im]  # needed later by postprocess to rescale boxes
    
    #PRE-PROCESS STEP A: pre-transform input img before inference
    with timed(store, "A_LetterBox", device):
        same_shapes = len({x.shape for x in im}) == 1 #determine if tensors are of same dimension/size
        letterbox = LetterBox( new_shape = (640,640), auto = same_shapes,  stride = 32 ) #TODO: might need to load stride from model itself
        resized_imgs = [letterbox(image = x) for x in im] #resize each image passed in the current batch

    #PRE-PROCESS STEP B: batch numpy stacking
    with timed(store, "B_batch_stack", device):
        if len(resized_imgs) == 1:
            im = torch.from_numpy(resized_imgs[0]).unsqueeze(0) 
        else:
            im = torch.from_numpy(np.stack(resized_imgs))

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

#heleper function for layer walk during inference stage, equivalent to _predict_once
def run_layers(model, layers, x, y):
    from m in layers:
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
        x = run_layers(model, model.model[: BACKBONE_BOUNDARY+1], x, y)
        #NOTE: loops through each layer interior to yolo until reaching the backbone stage's boundary
        #NOTE: potential inference step: SPATIAL POOLING which is considered separate from the backbone occurs
        #                        at layer 9 inside the model might want to pull that one out of backbone

    #INFERENCE STEP B: NECK fuse features across scales so both large and small objects have context
    #                  occurs in layers 10-21 of model
    with timed(store, "B_neck", device):
        x = run_layers(model, model.model[BACKBONE_BOUNDARY+1: NECK_BOUNDARY+1], x, y)
        for m in model.model[BACKBONE_BOUNDARY+1:NECK_BOUNDARY+1]: 
        #NOTE: loops through each layer interior to yolov8 until reaching the backbone stage's boundary
                
    #INFERENCE STEP D: HEAD predict bounding boxes and class scores from fused features
    #                  occurs in layer 22
    with timed(store, "C_head", device):
        x = run_layers(model, model.model[NECK_BOUNDARY+1:], x, y)
    return x


#based on postprocess function from ultralytics/models/yolo/detect/predict.py
def postprocess_stage(preds, im, orig_shapes, conf_thres: float = 0.25, iou_thres: float = 0.75, device = 'cuda:0'):
    store = timings["post_processing"]
    
    #POST PROCESS STEP A: apply NMS to get single best prediction for each detected object, i.e. no redundant bounding boxes
    with timed(store, "A_NMS", device):
        filtered_preds = non_max_suppression(   
            preds,
            conf_thres,
            iou_thres,
            classes = None,
            agnostic False,
            multi_label False,
            max_det 300,
            nc = 0,  # number of classes (optional)
            end2end = False,
            rotated = False,
        )
    
    #POST PROCESS STEP B: Rescale each img's boxes from letterboxed space to original image space  
    #Coordinate Rescaling using ultralytics/utils/ops.py
    with timed(store, "B_rescale_boxes",device):
        for pred, shape  zip(filtered_preds, orig_shapes):
            if len(pred):
                pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], shape) #maps coordinates back from 640x640 space to the original image shape

    return filtered_preds #returns the list of result objects containing the post processed predition


#based on benchmark class from ultralytics/engine/predictor.py
def run_pipeline(model, img_paths, batch_size: int = BATCH_SIZE, device = DEVICE, fp16 = FP16):

    #read images before beginning timing as ultralyitics doesn't include this portion in preprocess timing
    batches = [img_paths[i:i + batch_size] for i in range(0, len(img_paths), batch_size)] #fills batches with respective image paths
    loaded = [[cv2.imread(str(path)) for path in b] for b in batches] #loads file paths present in each batch

    #warmup model only on GPU
    if torch.device(device).type != "cpu":
        saved = {s: timings[s] for s in STAGES} #timing is hardcoded so need to send warmup times to garabge dict
        for s in STAGES: #loop through pre, inference, and post
            timings[s] = defaultdict(list)
            im, shapes = preprocess_stage(loaded[0], model, IMGSZ, device, fp16) #warmup with first loaded batch
            predictions = inference_stage(model, im, device)
            postprocess_stage(predictions, im, shapes, CONF_THRES, IOU_THRESH, device)

        for s in STAGES: #empty dictionary following warmup
            timings[s].clear()
            

    #run model on all batches and store results 
    results =[]
    num_images = 0

    for imgs in loaded:
        im, shapes = preprocess_stage(imgs, model, IMGSZ, device, fp16) #warmup with first loaded batch
        predictions = inference_stage(model, im, device)
        results.append(postprocess_stage(predictions, im, shapes, CONF_THRES, IOU_THRESH) );
        n_images += len(imgs) #keep track of number of images 


    #print results of runs
    print(f"per-image timings (ms), {n_images} images, batch={batch_size}")
        for stage in STAGES:
            total = 0.0
            for step, vals in timings[stage].items():
                per_img = sum(vals) / n_images
                total += per_img
                print(f"  {stage:<16} {step:<20} {per_img:8.2f}")
        print(f"  {stage:<16} {'TOTAL':<20} {total:8.2f}")
    return results    
    
    


def main(img_dir):
    #load model 
    #TODO: potential area to be expanded upon
    model = YOLO(WEIGHTS).model.fuse.eval().to(DEVICE)
        #.model exposes the NN layers within the model (enables differentiation btw backbone, neck, and head within inference)
        #.fuse fuses pairs of layers together, which YOLOv8n does
        #.eval() indicates use deployment, not training
        #.to(DEVICE) puts entire model on same DEVICE
    fp16 = FALSE

    #list of valid img_paths in test input directory
    formats  ={".jpg", ".jpeg", ".png"} #TODO: might want to add more acceptable file formats
    img_paths = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in formats)

    return run_pipeline(model, img_paths, BATCH_SIZE, DEVICE, fp16)

if __name__ == "__main__":
    main("~/tenstorrent2/YOLO/input_images")



