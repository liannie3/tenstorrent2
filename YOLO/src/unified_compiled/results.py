import csv #store timing and  results
from datetime import datetime #for displaying more accurate time stamp
import numpy as np

import config
from timing import timings

def _append_csv(path, fields, rows):
    path.parent.mkdir(parents = True, exist_ok = True)
    new_file = not path.exists() #indicates if we need to make a new file 
    with open(path, 'a', newline = "") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fields) #used for writing saved dictionary into csv
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

def display_results(stage_wall_timings, num_images, batch_size, device):
    pass_name = "substep_timing" if config.SUBSTEP_TIMING else "wall_timing"
    rows = [] #holds information to be stored later in store_results
    timestamp = datetime.now().isoformat(timespec = "seconds")
    print(f"per-image timings (ms), {num_images} images, batch={batch_size}, device={device}")
    for stage in config.STAGES: #for each stage
        total_time = 0.0
        for step, vals in timings[stage].items(): #loop through each stage's dictionary
            per_img = np.median(vals)/batch_size #TODO: could also use mean, but median is serviceable for 1 batch size
            total_time += per_img #track total time for stage
            rows.append( {"timestamp": timestamp, "script" : "stage_by_stage", "pass" : pass_name, "device" : device, "batch_size" : batch_size, "num_images" : num_images, 
                                          "stage" : stage, "step" :step, "ms" : per_img } )
            print(f"  {stage:<16} {step:<20} {per_img:8.2f}")
        if config.SUBSTEP_TIMING: #if substeps are timed
            rows.append( { "timestamp": timestamp, "script" : "stage_by_stage", "pass" : pass_name, "device" : device, "batch_size" : batch_size, "num_images" : num_images, 
                                            "stage" : stage, "step" : "SUM_OF_STEPS", "ms" : total_time } )
            print(f"  {stage:<16} {'SUM OF STEPS':<20} {total_time:8.2f}")
        else: #if only wall times are timed
            rows.append( { "timestamp": timestamp, "script" : "stage_by_stage", "pass" : pass_name, "device" : device, "batch_size" : batch_size, "num_images" : num_images, 
                                           "stage" : stage, "step" : "TOTAL_WALL_TIME", "ms" : np.median(stage_wall_timings[stage]) } )
            print(f"  {stage:<16} {'TOTAL WALL TIME':<20} {np.median(stage_wall_timings[stage]):8.2f}")
    return rows

def store_results(rows):
    _append_csv(config.CSV_PATH, config.CSV_FIELDS, rows)
    
def store_hw_window(monitor, phase, start, end):
    _append_csv(config.HARDWARE_PATH, config.HARDWARE_FIELDS,
                [{"monitor": monitor, "phase": phase, "start": start, "end": end}])

def store_hw_stages(runs):
    _append_csv(config.HARDWARE_STAGES_PATH, config.STAGE_FIELDS, runs)
