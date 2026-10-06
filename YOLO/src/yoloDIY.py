import cv2 #open source computer vision libray utilized by ultralytics
import torch
import time
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
from ultralytics.nn.tasks import attempt_load_weights #ultralyitics neural network function


#NOTE: run on CPU
from ultralytics.data.augment import LetterBox #ultralytics preprocess function to resize and pad
                                               #input images to a specified shape while preserving aspect ratio
                                               #additionally updates labels & bounding boxes respectively

from ultralytics.utils.nms import non_max_suppression #ultralyitics post process function that applies NMS on prediction results
                                                      #motivated to filter overlapping bounding boxes based on confidence and IoU (intersection over union) threshhold 

#NOTE: run on CPU                                   
from ultralytics.utils.ops import scale_boxes #ultralyitics utility function that rescales bounding boxes from one image to another

#to store timing for each layer of each stage
STAGES = ("pre_processing", "inference", "post_processing")
timings = {s: defaultdict(list) for s in STAGES} #create 


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
# 
def preprocess_stage(im: list[np.ndarray], device = 'cuda:0', fp16 = False):
    step_times = {} #hold internal step times
    
    orig_shapes = [x.shape for x in im]  # needed later by postprocess to rescale boxes
    
    #PRE-PROCESS STEP A: pre-transform input img before inference
    same_shapes = len({x.shape for x in im}) == 1 #determine if tensors are of same dimension/size
    letterbox = LetterBox( new_shape = (640,640), auto = same_shapes,  stride = 32 ) #stride logic might need to be addressed
    resized_imgs = [letterbox(image = x) for x in im] #resize each image passed in the current batch

    #PRE-PROCESS STEP B: batch numpy stacking

    if len(resized_imgs) == 1:
        im = torch.from_numpy(resized_imgs[0]).unsqueeze(0) 
    else:
        im = torch.from_numpy(numpy.stack(resized_imgs))

    #PRE-PROCESS STEP C: host to memory transfer batch transfer to gpu

    im = im.to(device) #tranfer tensors to gpu
    _sync(device) #wait for transfer to gpu

    #PRE-PROCESS STEP D: adjust and normalize tensor layout (GPU)

    im = im.permute(0, 3, 1, 2) #transfer from torch standard BHWC (batch, height, width, channels)
                                #to ultralytics standard BCHW (batch, channels, height, width) tensor dimension representation
    if im.shape[1] == 3:
        im = im.flip(1) # BGR to RGB if there are three channels
    im = im.contiguous()  #allocate a new contiguous (i.e. uninterrupted) block of memory and copy data into it
    if fp16:  
        im = im.half() #if fp16 (16 bit float), convert image pixels tensor from unint8 to fp16
    else:
        im = im.float()) #if not fp16 convert image pixels tensor from uint8 to fp32 (32 bit float)
    im = im.div_(255) #normalize tensor pixels from [0.0 - 255.0] -> [0.0 - 1.0]
    _sync(device) #wait for cuda kernels to finish loading
    
    return im, orig_shapes #return img tensor after pre processing is applied to it, and orignal tensor for post processing


#arg of model, feature map tensor im, and device that model is to be run on
#Based on _predict_once method from ultralytics/nn/tasks.py 
# NOTE: i think just iterates through the different layers present underneath the inference model
#       if further granularity is requested by saby we can maybe pinpoint which layer is associated with
#       which submodel? Similarly to how we identified spatial pooling to be the 9th layer
def inference_stage(model, im, device = 'cuda:0'):
    y = [] #store feature map tensors from earlier layers for future use
    x = im
    #INFERENCE STEP A: BACKBONE extract features from input at multiple resolutions
    #C2F (CSP (Constraint Satisfaction Problem) Bottleneck with 2 convolutions, faster)
    
    for m in model.model[: BACKBONE_BOUNDARY+1]: #NOTE: loops through each layer interior to yolov8
                                                 #until reaching the backbone stage's boundary
        #NOTE: tensor feature map shouldn't initially originate from  -1
        if m.f != -1: #.f indicates input came from (hence .f) model directly preeceding it
            # if isinstance(m.f,int): #if from is a single int not an int list
                # x = y[m.f] #grab stored output tensor from earlier layer to use as input for curr layer
            # else:
                # inputs = [] #create list to hold tensors c1 and c2
                # for j in m.f: #loop through all layers input came from
                    # if j == -1:
                        # inputs.append(x) #tensor from previous layer
                    # else:
                        # inputs.append(y[j]) #tensor from earlier layer
                # x = inputs #apply changes to feature map
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f] #does everything outlined above
        x = m(x) # run submodel with updated feature map
        if m.i in model.save: #model.save is a list of layer indiices that indicate which layers' outputs ust be saved
              y.append(x)
        else:
            y.append(None)
    _sync(device) #wait for gpu
    
        #NOTE: INFERENCE STEP B: SPATIAL POOLING which is considered separate from the backbone occurs
        #                        at layer 9 inside the model might want to pull that one out of backbone

    #INFERENCE STEP C: NECK fuse features across scales so both large and small objects have context
    #                  occurs in layers 10-21 of model
    
    for m in model.model[BACKBONE_BOUNDARY+1:NECK_BOUNDARY+1]: #NOTE: loops through each layer interior to yolov8
                                                               #until reaching the backbone stage's boundary
        #NOTE: tensor feature map shouldn't initially originate from  -1
        if m.f != -1: #.f indicates input came from (hence .f) model directly preeceding it
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x) # run submodel with updated feature map
        if m.i in model.save: #model.save is a list of layer indiices that indicate which layers' outputs ust be saved
              y.append(x)
        else:
            y.append(None)
    _sync(device) #wait for gpu

    #INFERENCE STEP D: HEAD predict bounding boxes and class scores from fused features
    #                  occurs in layer 22
    for m in model.model[NECK_BOUNDARY+1:]: #NOTE: loops through each layer interior to yolov8
                                            #until reaching the backbone stage's boundary
        #NOTE: tensor feature map shouldn't initially originate from  -1
        if m.f != -1: #.f indicates input came from (hence .f) model directly preeceding it
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x) # run submodel with updated feature map
        if m.i in model.save: #model.save is a list of layer indiices that indicate which layers' outputs ust be saved
              y.append(x)
        else:
            y.append(None)
    _sync(device) #wait for gpu    

    return x


#based on postprocess function from ultralytics/models/yolo/detect/predict.py
def postprocess_stage(preds, im, orig_shapes, conf_thres: float = 0.25, iou_thres: float = 0.75):

    #POST PROCESS STEP A: apply NMS to get single best prediction for each detected object, i.e. no redundant bounding boxes
    filtered_preds = non_max_suppression(   
        preds,
        conf_thres,
        iou_thres,
        classes = None,
        agnostic: bool = False,
        multi_label: bool = False,
        max_det: int = 300,
        nc: int = 0,  # number of classes (optional)
        max_time_img: float = 0.05,
        max_nms: int = 30000,
        max_wh: int = 7680,
        rotated: bool = False,
        end2end: bool = False,
        return_idxs: bool = False,
    )
    
    #POST PROCESS STEP B: Rescale each img's boxes from letterboxed space to original image space  
    #Coordinate Rescaling using ultralytics/utils/ops.py
    for pred, shape  zip(filtered_preds, orig_shapes):
        if len(pred):
            pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], shape) #maps coordinates back from 640x640 space to the original image shape

    return filtered_preds #returns the list of result objects containing the post processed predition


#based on benchmark class from ultralytics/engine/predictor.py
def run_pipeline(img_paths, batch_size: int = 8):

    #loop through all images present in the provided directory at a step size equal to batch size
    for i in range(0, len(image_paths), batch_size):
        preprocess_stage()

        inference_stage()

        postprocess_stage()
    


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #going to want to warmup the model (actual pipeline not just yolo model) 
    #might be useful to validate correctness of results against yolov8n.pt
    #setup phase separate of pre process

    dummy_tensor = preprocess_stage(dummy_imgs, device = device)

if __init__ == "__main__":
    main()


# # 1. SETUP (From ultralytics/models/yolo/model.py & engine/model.py)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = attempt_load_weights("yolov8n.pt", device=device)  # Loads raw PyTorch nn.Module
# model.eval()
# 
# # 2. SOURCE LOADING (From ultralytics/engine/predictor.py -> load_inference_source)
# orig_img = cv2.imread("image.jpg")  # Simulating raw source input (HWC, BGR)
# 
# 
# # PRE-PROCESSING STEP
# # FOLLOWING preprocessing func from ultralytics/models/yolo/detect/predict.py
# 
# # PREPROCESS STEP A: Pre-transform input image before before inference
# # Utilizing Letterbox func from (From ultralytics/data/augment.py)
# # recreating pre_transform wrapper func evoked by preprocessing
# img_resized = LetterBox(new_shape=(640, 640), auto=True, stride=32)(image=orig_img)
# 
# # Step B: Tensor formatting and normalization (From ultralytics/models/yolo/detect/predict.py)
# im = img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
# im = torch.from_numpy(im.copy()).to(device)
# im = im.float() / 255.0                      # Normalize pixel values to [0.0, 1.0]
# im = im[None]                                # Add batch dimension (BCHW)
# 
# # 4. INFERENCE (From ultralytics/engine/predictor.py -> inference)
# with torch.no_grad():
    # raw_preds = model(im) #pass to neural network
# 
# # 5. POST-PROCESSING (From ultralytics/models/yolo/detect/predict.py -> postprocess)
# # Step A: Non-Maximum Suppression filtering (From ultralytics/utils/ops.py)
# filtered_preds = non_max_suppression(raw_preds, conf_thres=0.25, iou_thres=0.45)
# 
# # Step B:
# for pred in filtered_preds:
    # if len(pred):
        # # Maps coordinates back from 640x640 space to the original image shape
        # pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], orig_img.shape).round()
# 
# # 6. RESULTS
# print("Processed Bounding Boxes [x1, y1, x2, y2, confidence, class]:")
# print(filtered_preds)
