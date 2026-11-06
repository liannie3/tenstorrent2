from pathlib import Path
import sys
import argparse #pass command line arguments to script to decide which runs to perform

from ultralytics import YOLO

import config
from hardware import hardware_run_pipeline, hardware_substages, identify_monitor
from run import run_pipeline, load_images

def load_model(fp16):
    model = YOLO(config.WEIGHTS).model.fuse().eval().to(config.DEVICE)
        #.model exposes the NN layers within the model (enables differentiation btw backbone, neck, and head within inference)
        #.fuse() fuses pairs of layers together, which YOLOv8n does
        #.eval() indicates use deployment, not training
        #.to(DEVICE) puts entire model on same DEVICE
    if(fp16):
        model.half()
    return model

def load_image_path(img_dir):
    #VALIDATE IMAGE PATH
    img_dir = Path(img_dir).expanduser().resolve()
    if not img_dir.exists():
        sys.exit(f"ERROR: IMAGE FOLDER NOT FOUND: {img_dir}")
    if not img_dir.is_dir():
        sys.exit(f"ERROR: PATH PROVIDED ISN'T A DIRECTORY: {img_dir}")
    #return file paths of only valid file types
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in config.IMG_FORMATS)
    
def main(setting, img_dir, batch_size : int = 1, fp16 : bool = False):
    img_paths = load_image_path(img_dir)
    model = load_model(fp16)
    results = None
        
    if setting in ("latency", "holistic", "profile"):
        print(f"   PASS 1: stage wall times only (synchronous profiling)   ")
        config.SUBSTEP_TIMING = False
        results = run_pipeline(model, img_paths, batch_size, config.DEVICE, fp16)
        if setting != "profile":
            config.SUBSTEP_TIMING = True
            print("   PASS 2: substep times (async with slight overhead from substep timers)   ")
            results = run_pipeline(model, img_paths, batch_size, config.DEVICE, fp16)
            config.SUBSTEP_TIMING = False
    if setting in ("hardware", "holistic"):
        found_monitor = identify_monitor()
        if found_monitor is None:
            print("  No hardware monitoring software found, skipping hardware benchmarking stages.")
        else:
            monitor_name = found_monitor[0] #name arg not cmd
            print(f"  PASS 3: {monitor_name} hardware metrics (full pipeline)")
            hardware_run_pipeline(model, img_paths * 5, batch_size, config.DEVICE, fp16)

            print(f"  PASS 4: stage based {monitor_name} hardware metrics (average of each pipeline stage)")
            images = load_images(img_paths)
            hardware_substages(model, images, config.DEVICE, fp16)
    
    return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("setting", choices = ["latency", "hardware", "holistic", "profile"] )
    parser.add_argument("batch_size", type = int, choices = [1, 2, 4, 8, 16] )
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    main(args.setting, config.IMG_DIR, args.batch_size, args.fp16)
