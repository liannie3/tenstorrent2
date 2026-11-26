"""YOLOv8n stage-by-stage benchmark on a Tenstorrent card (TT-NN), ported from the CUDA/Jetson version.
NOTE: the TT-NN model is tt-metal's yolov8x demo (v0.64.0, models/demos/yolov8x) ported to yolov8n, used to compare against
      the automatically compiled TT-XLA harness (xla_yolo)

  python main.py --setting verify                     step 1: run every stage on the card, check it against PyTorch on the CPU
  python main.py --setting latency --batch_size 4     step 2: wall pass, substep pass, end-to-end pass -> benchmarking/stage_timings_tt.csv
  python main.py --setting holistic                   both of the above

  every other setting (input size, weights dtype, math fidelity, device id) comes from config.py
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

import config


def load_model():
    #fp32 PyTorch model on the CPU: the weight source for the TT model and the verification reference
    model = YOLO(config.WEIGHTS).model.float().eval()
        #.model exposes the NN layers within the model (enables differentiation btw backbone, neck, and head within inference)
        #.float() the checkpoint ships fp16 weights
        #.eval() indicates use deployment, not training
        #NOTE: not .fuse(): the yolov8x preprocessor folds each BatchNorm into its conv itself, so it needs the .bn weights
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


def main(args):
    #ttnn is imported here so `python main.py --help` works on machines without tt-metal
    from run import load_images, run_pipeline
    from tt_device import TtContext, describe, open_device
    from ttnn_yolov8n import TtYolov8nModel
    from ttnn_yolov8n_utils import custom_preprocessor
    from verify import verify

    input_hw = tuple(config.INPUT_HW) #NOTE: should be (384, 640) to match the CUDA and TT-XLA letterbox

    img_paths = load_image_path(config.IMG_DIR)
    img_paths = img_paths[::args.step] #every nth image, same subset as the TT-XLA and CUDA harnesses' --step
    torch_model = load_model()
    images = load_images(img_paths)

    with open_device(config.DEVICE_ID) as device:
        ctx = TtContext(device)
        print(f"  device: {describe(device)}  |  {len(images)} images  |  input {input_hw[0]}x{input_hw[1]}")
        print(f"  TT-NN settings: batch {args.batch_size} | activations in L1 (sharded) | weights {config.WEIGHTS_DTYPE} | "
              f"math fidelity {config.MATH_FIDELITY}")
        #fold BatchNorm into the convs, split each C2f cv1 into its two halves, build the anchors for input_hw
        parameters = custom_preprocessor(device, torch_model.state_dict(), input_hw[0], input_hw[1],
                                         bfloat8=config.WEIGHTS_DTYPE == "bfp8")
        #NOTE: the TT model is built for one batch size and one input size (every conv is compiled for that shape)
        tt_model = TtYolov8nModel(device, parameters, res=input_hw, batch_size=args.batch_size,
                                  weights_dtype=config.WEIGHTS_DTYPE, math_fidelity=config.MATH_FIDELITY).detection_model
        results = None

        if args.setting in ("verify", "holistic"):
            print("   STEP 1: every stage on the card vs PyTorch CPU reference   ")
            verify(tt_model, torch_model, images, args.batch_size, ctx, input_hw,
                   img_names=[p.name for p in img_paths], save_images=False, num_batches=1)

        if args.setting in ("latency", "holistic"):
            print("   PASS 1: stage wall times (device synchronized between stages)   ")
            config.SUBSTEP_TIMING, config.SYNC_STAGES = False, True
            results = run_pipeline(tt_model, ctx, images, args.batch_size, input_hw, pass_name="wall_timing")

            print("   PASS 2: substep times (device synchronized around every substep)   ")
            config.SUBSTEP_TIMING, config.SYNC_STAGES = True, True
            results = run_pipeline(tt_model, ctx, images, args.batch_size, input_hw, pass_name="substep_timing")

            print("   PASS 3: end to end, no synchronization (host dispatch overlaps the card)   ")
            config.SUBSTEP_TIMING, config.SYNC_STAGES = False, False
            results = run_pipeline(tt_model, ctx, images, args.batch_size, input_hw, pass_name="e2e")
            config.SUBSTEP_TIMING, config.SYNC_STAGES = False, True
            print(f"  timings appended to {config.CSV_PATH}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=["verify", "latency", "holistic"], default="latency", help="what to run (default: %(default)s)")
    #NOTE: every feature map has to fit in L1 (the yolov8x demo's Conv2dL1FullSliceConfig), so large batches can run out of L1
    parser.add_argument("--batch_size", type=int, choices=[1, 2, 4, 8, 16, 32, 64], default=1, help="images per batch (default: %(default)s)")
    parser.add_argument("--step", type=int, choices=[1, 2, 4, 8, 16], default=1, help="use every nth image (default: %(default)s = all)")
    main(parser.parse_args())
