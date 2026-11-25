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
#backends like TensorRT for the same reason). The CUDA run used auto=True, which turns a 1600x900 CAM_FRONT frame into 384x640.
#Use --input-hw 384 640 to reproduce that shape on TT for an apples-to-apples comparison.
INPUT_HW = (384, 640) #match the CUDA and TT-XLA letterbox NOTE: USED TOBE 640 X 640
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
L1_SMALL_SIZE = 32768     #conv2d / max_pool2d keep their config tensors in the L1-small region; 0 makes them fail to allocate
TRACE_REGION_SIZE = 0     #no trace capture yet (that comes with the optimisation step)
NUM_COMMAND_QUEUES = 1

#where activations live between ops:
#  "dram": every intermediate is DRAM interleaved; convs take the DRAM-sliced path. Slowest but the hardest to break (step 1 default).
#  "l1":   intermediates are L1 interleaved (fits batch 1-2 at 640x640); faster, try it once "dram" verifies.
ACT_MEM = "dram"
WEIGHTS_DTYPE = "bf16"    #"bf16" or "bfp8" (block float 8, what tt-metal's YOLO demos use for speed; costs yolov8n accuracy)
MATH_FIDELITY = "hifi2"   #"lofi" (fastest), "hifi2", "hifi4" (most accurate)
FP32_DEST_ACC = True      #fp32 accumulation in the conv matmuls (HiFi2 + fp32 acc is what tt-metal's RT-DETR uses on Blackhole)

#True: DFL softmax + anchor decode + sigmoid run on the card (all of the Detect head on device, like ultralytics on GPU).
#False: the card returns raw head maps and the decode runs on the CPU in fp32 inside post-processing (fallback / accuracy reference).
DECODE_ON_DEVICE = True

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
PCC_PASS = 0.99
PCC_WARN = 0.97
