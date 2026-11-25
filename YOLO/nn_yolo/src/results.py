import csv #store timing and results
from datetime import datetime #for displaying more accurate time stamp

import numpy as np

import config
from timing import timings


def _append_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() #indicates if we need to make a new file
    with open(path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields) #used for writing saved dictionary into csv
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def display_results(pass_name, stage_wall_timings, e2e_timings, num_images, batch_size, meta):
    rows = [] #holds information to be stored later in store_results
    timestamp = datetime.now().isoformat(timespec="seconds")
    base = {"timestamp": timestamp, "script": "stage_by_stage_tt", "pass": pass_name, "device": meta["device"],
            "batch_size": batch_size, "num_images": num_images, "input_hw": meta["input_hw"], "act_mem": meta["act_mem"],
            "weights_dtype": meta["weights_dtype"], "math_fidelity": meta["math_fidelity"], "decode": meta["decode"]}
    print(f"per-image timings (ms), {num_images} images, batch={batch_size}, device={meta['device']}, input={meta['input_hw']}, "
          f"act_mem={meta['act_mem']}, weights={meta['weights_dtype']}, fidelity={meta['math_fidelity']}, decode={meta['decode']}")

    if pass_name == "e2e":
        per_img = float(np.median(e2e_timings)) / batch_size
        rows.append({**base, "stage": "all", "step": "TOTAL_E2E", "ms": per_img})
        print(f"  {'all':<16} {'TOTAL E2E':<20} {per_img:8.2f}   ({1000.0 / per_img:.1f} images/s)")
        return rows

    for stage in config.STAGES: #for each stage
        total_time = 0.0
        for step, vals in timings[stage].items(): #loop through each stage's dictionary
            per_img = float(np.median(vals)) / batch_size #median per batch, divided by the batch to get per image
            total_time += per_img #track total time for stage
            rows.append({**base, "stage": stage, "step": step, "ms": per_img})
            print(f"  {stage:<16} {step:<20} {per_img:8.2f}")
        if config.SUBSTEP_TIMING: #if substeps are timed
            rows.append({**base, "stage": stage, "step": "SUM_OF_STEPS", "ms": total_time})
            print(f"  {stage:<16} {'SUM OF STEPS':<20} {total_time:8.2f}")
        else: #if only wall times are timed
            #NOTE: divided by batch_size here too; the CUDA results.py printed this per batch under a per-image header
            wall = float(np.median(stage_wall_timings[stage])) / batch_size
            rows.append({**base, "stage": stage, "step": "TOTAL_WALL_TIME", "ms": wall})
            print(f"  {stage:<16} {'TOTAL WALL TIME':<20} {wall:8.2f}")
    return rows


def store_results(rows):
    _append_csv(config.CSV_PATH, config.CSV_FIELDS, rows)


def store_verify(rows):
    _append_csv(config.VERIFY_CSV_PATH, config.VERIFY_FIELDS, rows)
