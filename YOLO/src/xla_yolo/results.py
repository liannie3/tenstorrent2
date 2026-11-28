import csv #store timing and  results
from datetime import datetime #for displaying more accurate time stamp
import numpy as np

import config
from timing import timings

#helper function that returns the run settings stored with every latency row and shown in results header
#NOTE: step now stands for the step through rate of db of images and not substep stage name
def _run_info():
    precision = {"torch.float32": "fp32", "torch.float16": "fp16", "torch.bfloat16": "bf16"}.get(str(config.DTYPE), str(config.DTYPE))
    return {"step": config.STEP, "optimization": config.OPTIMIZATION, "precision": precision}
    
def _append_csv(path, fields, rows):
    path.parent.mkdir(parents = True, exist_ok = True)
    new_file = not path.exists() #indicates if we need to make a new file 
    with open(path, 'a', newline = "") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fields) #used for writing saved dictionary into csv
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

#NOTE: rows are now labelled "stage_by_stage_compiled" (was "stage_by_stage") so they can be told apart
#                from the eager harness's rows when both write to the same stage_timings.csv
def display_results(stage_wall_timings, num_images, batch_size, device):
    pass_name = "substep_timing" if config.SUBSTEP_TIMING else "wall_timing"
    rows = [] #holds information to be stored later in store_results
    timestamp = datetime.now().isoformat(timespec = "seconds")
    info = _run_info() #step, optimization and precision added to the end of every row
    print(f"timings (ms per batch, per image in parentheses), {num_images} images, batch={batch_size}, step={info['step']}, opt={info['optimization']}, precision={info['precision']}, device={device}")
    for stage in config.STAGES: #for each stage
        total_time = 0.0
        for substep, vals in timings[stage].items(): #loop through each stage's dictionary
            per_img = np.median(vals) / batch_size #TODO: could also use mean, but median is serviceable for 1 batch size
            total_time += per_img #track total time for stage
            rows.append( {"timestamp": timestamp, "script" : "stage_by_stage_compiled", "pass" : pass_name, "device" : device, "batch_size" : batch_size, "num_images" : num_images, 
                                          "stage" : stage, "substep" : substep, "ms" : per_img, **info } )
            print(f"  {stage:<16} {substep:<20} {per_img * batch_size:8.2f}   ({per_img:.2f} ms per image)")
        if config.SUBSTEP_TIMING: #if substeps are timed
            rows.append( { "timestamp": timestamp, "script" : "stage_by_stage_compiled", "pass" : pass_name, "device" : device, "batch_size" : batch_size, "num_images" : num_images, 
                                            "stage" : stage, "substep" : "SUM_OF_STEPS", "ms" : total_time, **info } )
            print(f"  {stage:<16} {'SUM OF STEPS':<20} {total_time * batch_size:8.2f}   ({total_time:.2f} ms per image)")
        else: #if only wall times are timed
            wall = np.median(stage_wall_timings[stage]) / batch_size #stage is timed per batch, divided so every row is per image like the substeps
            rows.append( { "timestamp": timestamp, "script" : "stage_by_stage_compiled", "pass" : pass_name, "device" : device, "batch_size" : batch_size, "num_images" : num_images, 
                                           "stage" : stage, "substep" : "TOTAL_WALL_TIME", "ms" : wall, **info } )
            print(f"  {stage:<16} {'TOTAL WALL TIME':<20} {wall * batch_size:8.2f}   ({wall:.2f} ms per image)")
    return rows
    
#end to end pass: no added syncs, total time of the whole run plus per batch spread
def display_e2e_results(batch_times, num_images, batch_size, device, start, end):
    timestamp = datetime.now().isoformat(timespec = "seconds")
    info = _run_info() #step, optimization and precision added to the end of every row
    num_batches = len(batch_times)
    total_per_batch = (end - start) * 1000 / num_batches   #whole run / batches, includes the one sync at the end
    median_per_batch = np.median(batch_times)
    p95_per_batch = np.percentile(batch_times, 95)
    print(f"end to end timings (ms per BATCH), {num_batches} batches of {batch_size}, step={info['step']}, opt={info['optimization']}, precision={info['precision']}, device={device}")
    print(f"  {'TOTAL / BATCHES':<20} {total_per_batch:8.2f}   ({total_per_batch / batch_size:.2f} ms per image)")
    print(f"  {'BATCH MEDIAN':<20} {median_per_batch:8.2f}   ({median_per_batch / batch_size:.2f} ms per image)")
    print(f"  {'BATCH P95':<20} {p95_per_batch:8.2f}   ({p95_per_batch / batch_size:.2f} ms per image)")
    rows = []
    for substep, ms in (("TOTAL_PER_BATCH", total_per_batch), ("BATCH_MEDIAN", median_per_batch), ("BATCH_P95", p95_per_batch)):
        rows.append( {"timestamp": timestamp, "script" : "stage_by_stage_compiled", "pass" : "e2e", "device" : device, "batch_size" : batch_size,
                      "num_images" : num_images, "stage" : "end_to_end", "substep" : substep, "ms" : ms, **info} )
    return rows

def store_results(rows):
    _append_csv(config.CSV_PATH, config.CSV_FIELDS, rows)
    
def store_hw_window(monitor, phase, batch_size, start, end, step, optimization):
    _append_csv(config.HARDWARE_PATH, config.HARDWARE_FIELDS,
                [{"monitor": monitor, "phase": phase, "batch_size" : batch_size, "start": start, "end": end,
                  "step": step, "optimization": optimization}]) #step / optimization stored so hw runs can be told apart

def store_hw_stages(runs):
    _append_csv(config.HARDWARE_STAGES_PATH, config.STAGE_FIELDS, runs)
