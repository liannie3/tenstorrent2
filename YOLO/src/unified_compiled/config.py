from pathlib import Path
import torch

#######################################################################################################################################################################  
#                        | | | |                                                                                                                                      #
#   File Paths           V V V V                                                                                                                                      #
#######################################################################################################################################################################  

BENCH_DIR = Path(__file__).parent.parent.parent / "benchmarking"
IMG_DIR = Path(__file__).parent.parent.parent / "input_images" / "CAM_FRONT"

CSV_PATH = BENCH_DIR / "stage_timings.csv" #computes path relative from the file to the folder containing it twice then to csv path 
HARDWARE_PATH = BENCH_DIR / "hw_metrics.csv"
HARDWARE_STAGES_PATH = BENCH_DIR / "hw_stages.csv"

#store tegrastats hardware results 
HARDWARE_LOG = CSV_PATH.parent / "tegrastats.log"

#######################################################################################################################################################################  
#                        | | | |                                                                                                                                      #
#   Fields               V V V V                                                                                                                                      #
#######################################################################################################################################################################  

#indicates fields to be saved in each row of csv dictionary
CSV_FIELDS = ["timestamp", "script", "pass", "device", "batch_size", "num_images", "stage", "step", "ms"] #nested dictionaries to asynchronously store timing for each layer of each stage
HARDWARE_FIELDS = ["monitor", "phase", "start", "end"]
STAGE_FIELDS = ["monitor", "stage", "start", "end", "iterations"]
IMG_FORMATS = [".jpg", ".jpeg", ".png"]

#pipeline stages
STAGES = ("pre_processing", "inference", "post_processing")

#######################################################################################################################################################################  
#                        | | | |                                                                                                                                      #
#   Model Settings       V V V V                                                                                                                                      #
#######################################################################################################################################################################  

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

#######################################################################################################################################################################  
#                        | | | |                                                                                                                                      #
#   Hardware monitor     V V V V                                                                                                                                      #
#######################################################################################################################################################################  

#indicates maximum of five seconds per stage for aggregate hardware testing
PHASE_LIMIT = 5.0 
HW_SAMPLE_SIZE = 25
#######################################################################################################################################################################  
#                        | | | |                                                                                                                                      #
#   Run flags            V V V V                                                                                                                                      #
#######################################################################################################################################################################  

#control logic for differntiating between run types
HARDWARE_METRICS = False #indicates hardware testing and no need for latency during run_pipeline    
SUBSTEP_TIMING = False #used to control if sub steps are timed, or pure wall times with syncs are timed
                       #necessary since calling record and query takes a little CPU time
                       #when measuring the substeps of each stage.
                       #ultralyitics times the more abstracted stages so this CPU time ins't incurred
                       #it shows up in latency numbers when CPU has to wait for GPU (i.e. CPU limited runs)
                       #with a more powerful CPU, this bottleneck wouldn't occur since GPU wouldn't be waiting for CPU (likely other way around)
                       #NOTE: right now wall-timing isn't async, but substep timing is 
                       #NOTE: evidence found in Profile class of ultralytics/utils/ops.py where they call accelerator.synchronize before measuring the perf_counter
PROFILE = False
