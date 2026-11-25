from pathlib import Path
import sys
import argparse #pass command line arguments to script to decide which runs to perform

from ultralytics import YOLO

import torch
import config
from hardware import hardware_run_pipeline, hardware_substages, identify_monitor
from run import run_pipeline, load_images
from stages import compile_stages #sets up the torch compiled stages

def load_model():
    model = YOLO(config.WEIGHTS).model.fuse().eval().to(config.DTYPE).to(config.DEVICE)
        #.model exposes the NN layers within the model (enables differentiation btw backbone, neck, and head within inference)
        #.fuse() fuses pairs of layers together, which YOLOv8n does
        #.eval() indicates use deployment, not training
        #.to(config.DTYPE) loads layer with data type format
        #.to(DEVICE) puts entire model on same DEVICE
    compile_stages(model) #wraps each GPU substep in torch.compile now that the model is final
    
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
    
def main(setting, img_dir, batch_size : int = 1, step : int = 1):
    img_paths = load_image_path(img_dir)
    img_paths = img_paths[::step] #NOTE: used for demo to exe time covering entire dataset
                                             #      every nth image, so a step of 4 runs 101/404 imgs through
                                             #      superior to simple subset of imgs since nu scences db are in order and some parts of db could be less dense w/ objs
    if not step == 1:
        print(f"   Using a subset of total {len(img_paths)} images (every {step} images from the test set)")
    else:
        print(f"   Using entire set of {len(img_paths)} test images")
    model = load_model()
    results = None

    if setting in ("e2e", "holistic"):
        print(f"    PASS 0: End to End latency (asynchronous capturing total host time per batch)  ")
        config.SUBSTEP_TIMING = False
        results = run_pipeline(model, img_paths, batch_size, config.DEVICE, "e2e")
    if setting in ("latency", "holistic", "profile"):
        print(f"   PASS 1: stage wall times only (ULTRALYITICS-style synchronous profiling)   ")
        config.SUBSTEP_TIMING = False
        results = run_pipeline(model, img_paths, batch_size, config.DEVICE)
        if setting != "profile":
            config.SUBSTEP_TIMING = True
            print("   PASS 2: substep times (async with slight overhead from substep timers)   ")
            results = run_pipeline(model, img_paths, batch_size, config.DEVICE)
            config.SUBSTEP_TIMING = False
            
    if setting in ("hardware", "holistic"):
        found_monitor = identify_monitor()
        if found_monitor is None:
            print("  No hardware monitoring software found, skipping hardware benchmarking stages.")
        else:
            monitor_name = found_monitor[0] #name arg not cmd
            print(f"  PASS 3: {monitor_name} hardware metrics (full pipeline)")
            hardware_run_pipeline(model, img_paths * 5, batch_size, config.DEVICE)

            print(f"  PASS 4: stage based {monitor_name} hardware metrics (average of each pipeline stage)")
            images = load_images(img_paths)
            hardware_substages(model, images, config.DEVICE)
    
    return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices = ["latency", "hardware", "holistic", "profile", "e2e"], default = "latency", help = "determines what setting harness runs under (latency, e2e, profile, hardware, or holistic)")
    parser.add_argument("--batch_size", type = int, choices = [1, 2, 4, 8, 16] , default = 1, help = "determines throughput of images sent through the model per batch (default to 1)")
    parser.add_argument("--precision", choices = ["bf16", "fp32"], default = "bf16", help = "determines model data percision (defaulted to bf16)") #model data precision, optional, bf16 default to best match TT
    parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image") #used to create accurate subset of db
    args = parser.parse_args()
    #NOTE: SETS DATA PRECISION BEFORE THE MODEL IS LOADED
    if args.precision == "fp32":
        config.DTYPE = torch.float32
    else:
        config.DTYPE = torch.bfloat16
    main(args.setting, config.IMG_DIR, args.batch_size, args.step)
