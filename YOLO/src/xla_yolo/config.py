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
CSV_FIELDS = ["timestamp", "script", "pass", "device", "batch_size", "num_images", "stage", "substep", "ms", "step", "optimization", "precision"] #nested dictionaries to asynchronously store timing for each layer of each stage
HARDWARE_FIELDS = ["monitor", "phase", "batch_size", "start", "end", "step", "optimization"]
STAGE_FIELDS = ["monitor", "stage", "batch_size", "start", "end", "iterations", "step", "optimization"]
IMG_FORMATS = [".jpg", ".jpeg", ".png"]

#pipeline stages
STAGES = ("pre_processing", "inference", "post_processing")

#######################################################################################################################################################################  
#                        | | | |                                                                                                                                      #
#   Model Settings       V V V V                                                                                                                                      #
#######################################################################################################################################################################  

WEIGHTS = "yolov8n.pt" 
DEVICE = "xla"
IMGSZ = 640 #640 px
BATCH_SIZE = 1 #set to 1 by default in ultralytics predict func
CONF_THRES = 0.25 #confindence threshold for NMS 
IOU_THRESH = 0.70 #intersection over union threshold for NMS
DTYPE = torch.bfloat16 #NOTE: no longer a cli arg, but bf16 (brain floating point 16) is bh p150a's native format
                       #      can support fp32 for output only, but no math within gpu can be done in fp32
                       #TODO: potentially change fp16 to bf16 on jetson for a better matched comparison


COMPILE_WARMUP = 2 #indicates the need for warming up on the first and second calls to compilation WITHOUT timing
                   #1st is the compilation of all graphs, 2nd is the recompilation of detect's head
                   #TODO: see if TT wants to compare compilation times between TT and CUDA
                   #NOTE: their current demos discard compilation times in analysis as well

#boundary layer indicies from ultralytics/cfg/models/v8/yolov8.yaml
BACKBONE_BOUNDARY = 9 #indicates backbone layers are 0-9 
NECK_BOUNDARY = 21 #indicates neck layers are 10-21.

#######################################################################################################################################################################  
#                        | | | |                                                                                                                                      #
#   Hardware monitor     V V V V                                                                                                                                      #
#######################################################################################################################################################################  

#TODO: add hw monitor once hardware metrics are implemented
#indicates maximum of five seconds per stage for aggregate hardware testing
PHASE_LIMIT = 5.0 
HW_SAMPLE_SIZE = 25
COOLDOWN = 30 #wait 30 seconds in between measuring hw substage activity, allows time for chip to cool and prevent heat from previous substage work from spilling over into another substage
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
                       #NOTE: jetson CUDA implementation substeps timed asynchronously with CUDA GPU events
                       #      PyTorch/XLA has no event equivalent, so a TT substep is timed by syncing before and after
                       #      Adding a host round trip per substep, so numbers include more sync overhead than CUDA's.
                       #      wall time pass is more equivalent
                       #TODO: can possibly write a small kernel that grabs GPU timestamp and use that in place of events
                       
PROFILE = False
E2E_FLUSH = False #True during the e2e pass, makes timing.timed() send each card substep's graph to the card at its end (see timing.py)
