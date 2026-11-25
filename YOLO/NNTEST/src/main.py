"""YOLOv8n stage-by-stage benchmark on a Tenstorrent card (TT-NN), ported from the CUDA/Jetson version.

  python main.py verify 1               step 1: run every substage on the card, check it against PyTorch on the CPU
  python main.py latency 1              step 2: wall pass, substep pass, end-to-end pass -> benchmarking/stage_timings_tt.csv
  python main.py holistic 1             both of the above (hardware metrics join this in step 3)
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

    config.TRACY_MARKERS = args.tracy
    input_hw = tuple(args.input_hw)
    if input_hw[0] % 32 or input_hw[1] % 32:
        sys.exit("ERROR: --input-hw must be multiples of 32")

    img_paths = load_image_path(args.img_dir)
    if args.num_images:
        img_paths = img_paths[: args.num_images]
    torch_model = load_model()
    images = load_images(img_paths)

    with open_device(args.device_id) as device:
        ctx = TtContext(device, act_mem=args.act_mem, weights_dtype=args.weights_dtype, math_fidelity=args.math_fidelity,
                        fp32_dest_acc=not args.no_fp32_acc)
        print(f"  device: {describe(device)}  |  {len(images)} images  |  input {input_hw[0]}x{input_hw[1]}")
        tt_model = TtYoloV8(torch_model, ctx, decode_on_device=args.decode == "device")
        results = None

        if args.setting in ("verify", "holistic"):
            print("   STEP 1: every substage on the card vs PyTorch CPU reference   ")
            verify(tt_model, torch_model, images, args.batch_size, ctx, input_hw,
                   img_names=[p.name for p in img_paths], save_images=args.save_images, num_batches=args.verify_batches)

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
    parser.add_argument("setting", choices=["verify", "latency", "holistic"])
    parser.add_argument("batch_size", type=int, choices=[1, 2, 4, 8, 16])
    parser.add_argument("--img-dir", default=str(config.IMG_DIR), help="folder of input images (default: %(default)s)")
    parser.add_argument("--num-images", type=int, default=None, help="only use the first N images")
    parser.add_argument("--input-hw", type=int, nargs=2, default=list(config.INPUT_HW), metavar=("H", "W"),
                        help="static letterbox size, multiples of 32 (default: %(default)s; 384 640 matches the CUDA run)")
    parser.add_argument("--act-mem", choices=["dram", "l1"], default=config.ACT_MEM,
                        help="where activations live between ops (default: %(default)s)")
    parser.add_argument("--weights-dtype", choices=["bf16", "bfp8"], default=config.WEIGHTS_DTYPE)
    parser.add_argument("--math-fidelity", choices=["lofi", "hifi2", "hifi4"], default=config.MATH_FIDELITY)
    parser.add_argument("--no-fp32-acc", action="store_true", help="accumulate conv matmuls in bf16 instead of fp32")
    parser.add_argument("--decode", choices=["device", "host"], default="device" if config.DECODE_ON_DEVICE else "host",
                        help="run the box decode on the card or on the CPU (default: %(default)s)")
    parser.add_argument("--device-id", type=int, default=config.DEVICE_ID)
    parser.add_argument("--verify-batches", type=int, default=1, help="how many batches verify compares (default: 1)")
    parser.add_argument("--save-images", action="store_true", help="verify: write TT detections drawn on the images")
    parser.add_argument("--tracy", action="store_true", help="emit Tracy zones/signposts (use under `python -m tracy`)")
    main(parser.parse_args())
