from pathlib import Path

#######################################################################################################################################################################
#                        | | | |                                                                                                                                      #
#   File Paths           V V V V                                                                                                                                      #
#######################################################################################################################################################################

#same layout as the CUDA version: <repo>/benchmarking and <repo>/input_images/CAM_FRONT, three levels up from this file
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = ROOT_DIR / "benchmarking"
IMG_DIR = ROOT_DIR / "input_images" / "CAM_FRONT"

#TT results go to their own csv so the extra TT columns never collide with the CUDA stage_timings.csv header
CSV_PATH = BENCH_DIR / "stage_timings_tt.csv"
VERIFY_CSV_PATH = BENCH_DIR / "verify_tt.csv"
VERIFY_IMG_DIR = BENCH_DIR / "tt_verify"  #annotated TT detections are written here by `main.py verify --save-images`

#######################################################################################################################################################################
#                        | | | |                                                                                                                                      #
#   Fields               V V V V                                                                                                                                      #
#######################################################################################################################################################################

#CUDA fields + the TT settings that change the numbers (memory placement, weight dtype, math fidelity, where the box decode runs)
CSV_FIELDS = ["timestamp", "script", "pass", "device", "batch_size", "num_images", "input_hw", "act_mem", "weights_dtype",
              "math_fidelity", "decode", "stage", "step", "ms"]
VERIFY_FIELDS = ["timestamp", "device", "batch_size", "input_hw", "act_mem", "weights_dtype", "math_fidelity", "decode",
                 "check", "pcc", "status"]
IMG_FORMATS = [".jpg", ".jpeg", ".png"]
#pipeline stages
STAGES = ("pre_processing", "inference", "post_processing")

#######################################################################################################################################################################
#                        | | | |                                                                                                                                      #
#   Model Settings       V V V V                                                                                                                                      #
#######################################################################################################################################################################

WEIGHTS = "yolov8n.pt"
#TT kernels are compiled for one static shape, so the letterbox always pads to INPUT_HW (ultralytics uses auto=False for static
#backends like TensorRT for the same reason). For a 1600x900 CAM_FRONT frame this gives the same 384x640 tensor as the
#auto=True letterbox in the CUDA and TT-XLA harnesses
INPUT_HW = (384, 640) #(height, width) NOTE: tt-metal's yolov8x demo only ran 640x640
BATCH_SIZE = 1
CONF_THRES = 0.25 #confidence threshold for NMS
IOU_THRESH = 0.70 #intersection over union threshold for NMS

#boundary layer indicies from ultralytics/cfg/models/v8/yolov8.yaml
BACKBONE_BOUNDARY = 9 #indicates backbone layers are 0-9
NECK_BOUNDARY = 21 #indicates neck layers are 10-21.

#######################################################################################################################################################################
#                        | | | |                                                                                                                                      #
#   Tenstorrent device   V V V V                                                                                                                                      #
#######################################################################################################################################################################

DEVICE_ID = 0             #p150a is a single chip, so this is almost always 0 (check with `tt-smi`)
L1_SMALL_SIZE = 24576     #YOLOV8X_L1_SMALL_SIZE from tt-metal's yolov8x demo (conv2d / max_pool2d config tensors live here)
TRACE_REGION_SIZE = 0     #no trace capture (skipped on both sides, like CUDA graphs on the Jetson)
NUM_COMMAND_QUEUES = 1

#NOTE: activations stay in L1 (height/block sharded) between ops, exactly like the yolov8x demo, so there is no DRAM option.
#      Every conv uses Conv2dL1FullSliceConfig (the whole feature map must fit in L1), which caps the batch size
WEIGHTS_DTYPE = "bfp8"    #"bfp8" (yolov8x demo default: bfloat8_b weights AND activations) or "bf16" (more accurate, slower)
MATH_FIDELITY = "lofi"    #"lofi" (yolov8x demo default, fastest), "hifi2", "hifi4" (most accurate)
STEM_DRAM_SLICE = True    #True: the first conv reads its input from DRAM in slices (needed above batch ~4 at 384x640)
                          #False: exactly like the yolov8x demo (whole input in L1); fine at batch 1-4, a bit faster

#######################################################################################################################################################################
#                        | | | |                                                                                                                                      #
#   Run flags            V V V V                                                                                                                                      #
#######################################################################################################################################################################

#control logic for differentiating between run types (same roles as the CUDA version)
SUBSTEP_TIMING = False    #True: synchronize the device around every substep and time each one (pass 2)
SYNC_STAGES = True        #True: synchronize the device between stages (pass 1, ultralytics Profile style). False: pass 3 (end to end)
                          #NOTE: TT has no CUDA-event equivalent with timestamps, so substep timing here is sync + perf_counter.
                          #      Device-side per-op times come from the tt-metal profiler (Tracy), which is a later step.
TRACY_MARKERS = False     #push Tracy zones around stages/substeps (the NVTX equivalent); only useful under `python -m tracy`
WARMUP_ITERS = 2          #first iteration JIT-compiles kernels and prepares conv weights on device; second checks steady state

#verification thresholds (Pearson correlation vs the PyTorch CPU reference)
#NOTE: tt-metal's yolov8x demo validated its whole-model output at PCC 0.978 with bfp8 + LoFi, so 0.97 is a WARN, not a FAIL
PCC_PASS = 0.99
PCC_WARN = 0.97
