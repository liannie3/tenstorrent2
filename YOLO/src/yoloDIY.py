import cv2 #open source computer vision libray utilized by ultralytics
import torch
import time
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
from ultralytics import YOLO
from pathlib import Path
#NOTE: runs on CPU
from ultralytics.data.augment import LetterBox #ultralytics preprocess function to resize and pad
                                               #input images to a specified shape while preserving aspect ratio
                                               #additionally updates labels & bounding boxes respectively

from ultralytics.utils.nms import non_max_suppression #ultralyitics post process function that applies NMS on prediction results
                                                      #motivated to filter overlapping bounding boxes based on confidence and IoU (intersection over union) threshhold 

#NOTE: runs on CPU                                   
from ultralytics.utils.ops import scale_boxes #ultralyitics utility function that rescales bounding boxes from one image to another
import torchvision #forces ultralytics' NMS to use torchvision's GPU based NMS

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
IOU_THRESH = 0.70 #intersection over union threshold for NMS
FP16 = False
#boundary layer indicies from ultralytics/cfg/models/v8/yolov8.yaml
BACKBONE_BOUNDARY = 9 #indicates backbone layers are 0-9 
NECK_BOUNDARY = 21 #indicates neck layers are 10-21.

def _sync(device): #only waits  for transfer to gpu if we're using gpu in the first place
    if torch.device(device).type == "cuda": 
        torch.cuda.synchronize()
    #NOTE: needs to be adjusted for TT
    
SUBSTEP_TIMING = False #used to compare the wall times with or without the substep timings
                       #necessary since calling record and query takes a little CPU time
                       #when measuring the substeps of each stage.
                       #ultralyitics times the more abstracted stages so this CPU time ins't incurred
                       #it shows up in latency numbers when CPU has to wait for GPU (i.e. CPU limited runs)
                       #with a more powerful CPU, this bottleneck wouldn't occur since GPU wouldn't be waiting for CPU (likely other way around)
                       
async_timings = [] #GPU event pairs that are waiting to be read
#timer helper
#sync's before and after to isolate current step's GPU work (i.e. no spill over from previous step)
#runs setup code before the block and runs exit code following the block
#NOTE: CHANGED FROM SYNCHRONOUS TO ASYNCHRONOUS TIMING
#TODO: WILL HAVE TO ADJUST OR CREATE NEW VERSION FOR TT
@contextmanager
def timed(store, name, device, gpu_work = True):
    if not SUBSTEP_TIMING:
        yield
        return
    if torch.device(device).type == "cuda" and gpu_work:
        start = torch.cuda.Event(enable_timing = True)
        end = torch.cuda.Event(enable_timing = True)
        start.record() #gpu timestamp this point wihtout cpu waiitng
        yield
        end.record()
        end.query() #tells driver to send out queued events now instead of waiting till they're done processing
                    #NOTE: intended to make more realistic to what occurs in actual yolo package
        async_timings.append((store, name, start, end)) #store timing data to be processed after run
    else:
        t0 = time.perf_counter()
        yield  #exit code below 
        store[name].append((time.perf_counter() - t0 ) * 1000) #convert to ms timing and store in dictionary at associated key

#TODO: WILL HAVE TO ADJUST OR CREATE NEW VERSION FOR TT
def process_timings(): #reads all pending GPU events achronously at end of run
    if async_timings: #if events are pending
        torch.cuda.synchronize() #force CPU to wait for GPU
        for store, name, start, end in async_timings: #store all GPU event data
            store[name].append(start.elapsed_time(end)) 
        async_timings.clear() #empty for subsequent runs


def display_results(stage_wall_timings,num_images, batch_size, device):
    print(f"per-image timings (ms), {num_images} images, batch={batch_size}, device={device}")
    for stage in STAGES: #for each stage
        total_time = 0.0
        for step, vals in timings[stage].items(): #loop through each stage's dictionary
            per_img = np.median(vals) #TODO: could also use mean, but median is serviceable for 1 batch size
            total_time += per_img #track total time for stage
            print(f"  {stage:<16} {step:<20} {per_img:8.2f}")
        if SUBSTEP_TIMING: #if we are including substep timing
            print(f"  {stage:<16} {'SUM OF STEPS':<20} {total_time:8.2f}")
            print(f"  {stage:<16} {'TOTAL WALL TIME':<20} {np.median(stage_wall_timings[stage]):8.2f}")
        else: #if we aren't including substep timing
            print(f"  {stage:<16} {'TOTAL WALL TIME':<20} {np.median(stage_wall_timings[stage]):8.2f}")
            
# def record_results():

#######################################################################################################################################################################  
#                        | | | |
#   Pre-Processing Stage V V V V
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
        letterbox = LetterBox( new_shape = (IMGSZ,IMGSZ), auto = same_shapes,  stride = 32 ) #TODO: might need to load stride from model itself
        resized_imgs = [letterbox(image = x) for x in im] #resize each image passed in the current batch

    #PRE-PROCESS STEP B: batch numpy stacking
    #NOTE: occurs before host to device (GPU) transfer so not GPU work
    with timed(store, "B_batch_stack", device, gpu_work = False):
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
########################################################################################################################################################################
#                   | | | |
#   Inference Stage V V V V
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
        x = run_layers(model, model.model[: BACKBONE_BOUNDARY+1], x, y)
        #NOTE: loops through each layer interior to yolo until reaching the backbone stage's boundary
        #NOTE: potential inference step: SPATIAL POOLING which is considered separate from the backbone occurs
        #                        at layer 9 inside the model might want to pull that one out of backbone

    #INFERENCE STEP B: NECK fuse features across scales so both large and small objects have context
    #                  occurs in layers 10-21 of model
    with timed(store, "B_neck", device):
        x = run_layers(model, model.model[BACKBONE_BOUNDARY+1: NECK_BOUNDARY+1], x, y)
        #NOTE: loops through each layer interior to yolov8 until reaching the backbone stage's boundary
                
    #INFERENCE STEP D: HEAD predict bounding boxes and class scores from fused features
    #                  occurs in layer 22
    with timed(store, "C_head", device):
        x = run_layers(model, model.model[NECK_BOUNDARY+1:], x, y) #runs HEAD layer only
    return x
########################################################################################################################################################################
#                         | | | |
#   Post-Processing Stage V V V V
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

    return filtered_preds #returns the list of result objects containing the post processed predition
########################################################################################################################################################################


@torch.inference_mode() #indicates not training, for realistic performance
#based on benchmark class from ultralytics/engine/predictor.py
def run_pipeline(model, img_paths, batch_size: int = BATCH_SIZE, device = DEVICE, fp16 = FP16):
    #reset async timigns from previous runs
    for s in STAGES:
        timings[s].clear()
    async_timings.clear()
    
    #read images before beginning timing as ultralyitics doesn't include this portion in preprocess timing
    images = []
    for path in img_paths:
        img = cv2.imread(str(path))
        if img is None: #cv2 returns None in case of error
            print(f"WARNING: SKIPPING UNREADABLE PATH: {path}")
            continue
        images.append(img)
    if not images:
        sys.exit("ERROR: NO IMAGES FROM DIRECTORY COULD BE READ")
        
    loaded = [images[i:i+batch_size] for i in range(0, len(images), batch_size)] #loads file paths present in each batch

    #warmup model only on GPU
    #TODO: verify how YOLO warmsup model
    if torch.device(device).type != "cpu": #NOTE: if intend to begin in steady state CPU for small img sample size, warmup CPU too (YOLOv8 doesn't do this)
        im, shapes = preprocess_stage(loaded[0], device, fp16) #warmup with first loaded batch
        predictions = inference_stage(model, im, device)
        postprocess_stage(predictions, im, shapes, CONF_THRES, IOU_THRESH, device)
        for s in STAGES: #empty dictionary following warmup
            timings[s].clear()
        async_timings.clear() #clear pending GPU events from warmup
    
    #run model on all batches and store results 
    results =[]
    num_images = 0

    #NOTE: internal timing of model stages only times either GPU or CPU work, when in reality both are occuring during some steps
    #      for instance during inference there are tiny overhead operations that queue on the CPU, and the GPU has to wait for them
    #      to most accurately represent Yolov8n's timing we'll keep track of both time on GPU/CPU and wall-time (or real time) and display both for extra context
    stage_wall_timings = {s : [] for s in STAGES} #wall clock time to be hollistically measured like ultralytics does
    for imgs in loaded:
        _sync(device) #isolates work between stages
        t0 = time.perf_counter()
        im, shapes = preprocess_stage(imgs, device, fp16)
        _sync(device)
        t1 = time.perf_counter() 
        predictions = inference_stage(model, im, device)
        _sync(device)
        t2 = time.perf_counter()
        results.append(postprocess_stage(predictions, im, shapes, CONF_THRES, IOU_THRESH, device) );
        _sync(device)
        t3 = time.perf_counter()
        for s, dt in zip(STAGES, (t1-t0, t2-t1, t3-t2) ):
            stage_wall_timings[s].append(dt * 1000)
        num_images += len(imgs) #keep track of number of images 
    process_timings()
    
    #print results of runs
    display_results(stage_wall_timings, num_images, batch_size, device)
    return results


# #NOTE: temporary function to gauge the latency incurred by overhead python wrapper calls
# def bench_wrappers(model, img, device = DEVICE, fp16 = FP16):
#     SUBSTEP_TIMING = False
#     #compares the raw layer loop vs model(im) vs ultralytics' full inference path on one image
#     y = YOLO(WEIGHTS)
#     y.predict(img, device = device, verbose = False) #builds ultralytics' predictor + AutoBackend
#     im, _ = preprocess_stage([img], device, fp16)
# 
#     def bench(f, n = 300):
#         for _ in range(20): f() #warmup
#         _sync(device); t = time.perf_counter()
#         for _ in range(n): f()
#         _sync(device); return (time.perf_counter() - t) / n * 1000
# 
#     tests = {
#             "your layer loop ": lambda: run_layers(model, model.model, im, []),
#             "model(im)       ": lambda: model(im),
#             "ultralytics path": lambda: y.predictor.inference(im),
#         }
#     names = list(tests)
#     with torch.inference_mode():
#         for r in range(3):
#             order = names[r:] + names[:r] #rotate which one goes first each round
#             print(f"round {r + 1}: " + ", ".join(f"{n.strip()} {bench(tests[n]):.2f} ms" for n in order))


def main(img_dir):
    #VALIDATE IMAGE PATH
    img_dir = Path(img_dir).expanduser().resolve()
    if not img_dir.exists():
        sys.exit(f"ERROR: IMAGE FOLDER NOT FOUND: {img_dir}")
    if not img_dir.is_dir():
        sys.exit(f"ERROR: PATH PROVIDED ISN'T A DIRECTORY: {img_dir}")
    
    #load model
    #TODO: potential area to be expanded upon
    model = YOLO(WEIGHTS).model.fuse().eval().to(DEVICE)
        #.model exposes the NN layers within the model (enables differentiation btw backbone, neck, and head within inference)
        #.fuse() fuses pairs of layers together, which YOLOv8n does
        #.eval() indicates use deployment, not training
        #.to(DEVICE) puts entire model on same DEVICE
    fp16 = False #NOTE: if we ever need to call model on fp16 then make sure to set model.half() accordingly

    #list of valid img_paths in test input directory
    formats  ={".jpg", ".jpeg", ".png"} #TODO: might want to add more acceptable file formats
    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in formats) #grab file paths of only valid file types

    global SUBSTEP_TIMING #initially false
    print("   PASS 1: stage wall times only (overhead from substep timers disabled)   ")
    results = run_pipeline(model, img_paths, BATCH_SIZE, DEVICE, fp16)

    SUBSTEP_TIMING = True
    print("   PASS 2: stage wall times and substep times (slight overhead from substep timers)   ")
    results = run_pipeline(model, img_paths, BATCH_SIZE, DEVICE, fp16)
    # bench_wrappers(model, cv2.imread(str(img_paths[0])))
    return results




if __name__ == "__main__":
    main(Path(__file__).parent.parent/ "input_images" / "CAM_FRONT")
