from pathlib import Path

#PATHS
HARNESS_DIR = Path(__file__).parent.parent.parent / "xla_yolo" #dir with the TT-XLA harness main.py
BENCH_DIR = Path(__file__).parent.parent.parent.parent / "benchmarking"  #dir with stage_timings.csv, profile reports saved here too

#TRACY SIGNPOSTS (run.py / timing.py emit "<name>_start" and "<name>_end")
STAGES = ("preprocess", "inference", "postprocess") #yolo stage signposts
TIMED_LOOP = "timed_batch_loop" #only signposts / ops inside the e2e batch loop are used
E2E_STEPS = ("BATCH_MEDIAN", "TOTAL_PER_BATCH") #e2e rows in stage_timings.csv to use, first one found wins

#TRACY OPTIONS
#-r: build the ops report csv (one row per card op with its device duration, plus one row per signpost)
#--op-support-count: how many ops the on-card profiler buffer keeps; the default is too small for a whole e2e run
TRACY_FLAGS = ["-r", "--op-support-count", "20000"]
