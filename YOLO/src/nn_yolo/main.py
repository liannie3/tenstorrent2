"""YOLOv8n stage-by-stage benchmark on a Tenstorrent card (TT-NN), ported from the CUDA/Jetson version.
NOTE: hand-written TT-NN version of YOLOv8n, used to compare against the automatically compiled TT-XLA harness (xla_yolo)

  python main.py --setting verify                     step 1: run every substage on the card, check it against PyTorch on the CPU
  python main.py --setting latency --batch_size 16    step 2: wall pass, substep pass, end-to-end pass -> benchmarking/stage_timings_tt.csv
  python main.py --setting holistic                   both of the above
  python main.py --setting latency --act-mem l1       activations kept in on-chip SRAM (closest match to TT-XLA optimization level 2)

  every other setting (input size, weights dtype, math fidelity, decode location, device id) comes from config.py
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

import config


def load_model():
    #fused fp32 PyTorch model on the CPU: the weight source for the TT model and the verification reference
    model = YOLO(config.WEIGHTS).model.fuse().eval()
        #.model exposes the NN layers within the model (enables differentiation btw backbone, neck, and head within inference)
        #.fuse() folds each BatchNorm into its conv, which is also what the TT convs use as weights
        #.eval() indicates use deployment, not training
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
    from tt_yolo import TtYoloV8
    from verify import verify

    input_hw = tuple(config.INPUT_HW) #NOTE: should be (384, 640) to match the CUDA and TT-XLA letterbox

    img_paths = load_image_path(config.IMG_DIR)
    img_paths = img_paths[::args.step] #every nth image, same subset as the TT-XLA and CUDA harnesses' --step
    torch_model = load_model()
    images = load_images(img_paths)

    with open_device(config.DEVICE_ID) as device:
        ctx = TtContext(device, act_mem=args.act_mem, weights_dtype=config.WEIGHTS_DTYPE, math_fidelity=config.MATH_FIDELITY,
                        fp32_dest_acc=True) #accumulate conv matmuls in fp32
        print(f"  device: {describe(device)}  |  {len(images)} images  |  input {input_hw[0]}x{input_hw[1]}")
        print(f"  TT-NN settings: batch {args.batch_size} | activations in {args.act_mem} | weights {config.WEIGHTS_DTYPE} | "
              f"math fidelity {config.MATH_FIDELITY}")
        tt_model = TtYoloV8(torch_model, ctx, decode_on_device=config.DECODE_ON_DEVICE)
        results = None

        if args.setting in ("verify", "holistic"):
            print("   STEP 1: every substage on the card vs PyTorch CPU reference   ")
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
    parser.add_argument("--batch_size", type=int, choices=[1, 2, 4, 8, 16, 32, 64], default=1, help="images per batch (default: %(default)s)")
    parser.add_argument("--step", type=int, choices=[1, 2, 4, 8, 16], default=1, help="use every nth image (default: %(default)s = all)")
    parser.add_argument("--act-mem", choices=["dram", "l1"], default=config.ACT_MEM,
                        help="where activations live between ops: dram, or l1 = on-chip SRAM (default: %(default)s)")
    main(parser.parse_args())
