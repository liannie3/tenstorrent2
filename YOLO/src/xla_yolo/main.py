from pathlib import Path
import sys
import argparse #pass command line arguments to script to decide which runs to perform

from ultralytics import YOLO
import torch_xla.runtime as xr #used to pick the bh p150a card as the xla (i.e. accelerated linear algebra) device

import config
from hardware import hardware_run_pipeline, hardware_substages, identify_monitor

from run import run_pipeline, load_images
from stages import compile_stages #sets up the torch.compiled gpu substages

def load_model():
    model = YOLO(config.WEIGHTS).model.fuse().eval().to(config.DTYPE).to(config.DEVICE)
        #.model exposes the NN layers within the model (enables differentiation btw backbone, neck, and head within inference)
        #.fuse() fuses pairs of layers together, which YOLOv8n does
        #.eval() indicates use deployment, not training
        #NOTE: TT difference .to(DTYPE) converts the weights to bf16 on the CPU, 
        #NOTE: then .to(DEVICE) copies them into the card's DRAM
    compile_stages(model) #wraps each GPU substep in torch.compile once the model is configured (fused, eval, on DEVICE, in DTYPE)
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
    
def main(setting, img_dir, batch_size : int = 1):
    xr.set_device_type("TT") #pytorch xla defaults to run on host as device, so select TT card as device before model
    img_paths = load_image_path(img_dir)
    model = load_model()
    results = None
        
    if setting in ("latency", "holistic", "profile"):
        print(f"   PASS 1: stage wall times only (synchronous profiling)   ")
        config.SUBSTEP_TIMING = False
        results = run_pipeline(model, img_paths, batch_size, config.DEVICE)
        if setting != "profile":
            config.SUBSTEP_TIMING = True
            print("   PASS 2: substep times (card synced around every card substep, see timing.py)   ")
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
    parser.add_argument("setting", choices = ["latency", "profile", "holistic", "hardware"] )
    parser.add_argument("batch_size", type = int, choices = [1, 2, 4, 8, 16] )
    args = parser.parse_args()
    main(args.setting, config.IMG_DIR, args.batch_size)
