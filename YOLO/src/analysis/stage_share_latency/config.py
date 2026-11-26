from pathlib import Path

#PATHS
HARNESS_DIR = Path(__file__).parent.parent.parent / "substage_compiled" #dir with main.py
BENCH_DIR = Path(__file__).parent.parent.parent.parent / "benchmarking"  #dir with stage_timings.csv, profile reports saved here too

#NVTX MARKERS
STAGES = ("preprocess", "inference", "postprocess") #yolo stage markers
TIMED_LOOP = "timed_batch_loop" #only ranges inside the e2e batch are used
E2E_STEPS = ("BATCH_MEDIAN", "TOTAL_PER_BATCH") #e2e rows in stage_timings.csv to use, first one found wins
