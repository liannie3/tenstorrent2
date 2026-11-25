from pathlib import Path #allows for finding path of input images
from ultralytics import YOLO
import cv2, sys, numpy as np
import argparse #cli args

IMG_DIR = Path(__file__).parent.parent.parent / "input_images" / "CAM_FRONT" #YOLO/input_images/CAM_FRONT, from wherever this is run
IMG_FORMATS = [".jpg", ".jpeg", ".png"] #same as config.IMG_FORMATS

parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--step", type = int, choices = [1,2,4,8,16], default = 1, help = "creates subset of image database composed of every nth image (defaulted to 1)") #used to create accurate subset of db
args = parser.parse_args()

#same format of validating then loading image path in main.py
def load_image_path(img_dir):
    #VALIDATE IMAGE PATH
    img_dir = Path(img_dir).expanduser().resolve()
    if not img_dir.exists():
        sys.exit(f"ERROR: IMAGE FOLDER NOT FOUND: {img_dir}")
    if not img_dir.is_dir():
        sys.exit(f"ERROR: PATH PROVIDED ISN'T A DIRECTORY: {img_dir}")
    #return file paths of only valid file types
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_FORMATS)

imgs = [cv2.imread(str(p)) for p in load_image_path(IMG_DIR)[::args.step]] #read each image in the subset created by step
model = YOLO("yolov8n.pt")
#Use settings as config.py: 640 px, conf 0.25, IoU 0.7, GPU 0, fp32 (half=True for fp16)
#NOTE: add compilation to match updated graph based harness
preds = [model.predict(im, imgsz = 640, conf = 0.25, iou = 0.7, device = 0, compile = True, verbose = False, rect = True)[0] for im in imgs]
warmup = 5 #skip the first few calls as warmup
print(f"{len(preds) - warmup} images, median ms per image:")
for k in ("preprocess", "inference", "postprocess"):
    print(f"  {k:<12} {np.median([r.speed[k] for r in preds[warmup:]]):.2f}") #take median latency of each stage as results.py does
                                                                            #NOTE: r.speed contains the per img ms latency from ultralyitics native Profile func
