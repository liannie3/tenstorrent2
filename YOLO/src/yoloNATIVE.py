from pathlib import Path #allows for finding path of input images
from ultralytics import YOLO
import cv2, glob, numpy as np

img_dir = Path(__file__).parent.parent / "input_images" / "CAM_FRONT" #YOLO/input_images/CAM_FRONT, from wherever this is run
model = YOLO("yolov8n.pt")
#Use settings as config.py: 640 px, conf 0.25, IoU 0.7, GPU 0, fp32 (half=True for fp16)
#NOTE: add compilation to match updated graph based harness
preds = [model.predict(im, imgsz = 640, conf = 0.25, iou = 0.7, device = 0, half = False, compile = True, verbose = False)[0] for im in imgs]
warm = 5 #skip the first few calls as warmup
print(f"{len(preds) - warm} images, median ms per image:")
for k in ("preprocess", "inference", "postprocess"):
    print(f"  {k:<12} {np.median([r.speed[k] for r in preds[warm:]]):.2f}") #take median latency of each stage as results.py does
                                                                            #NOTE: r.speed contains the per img ms latency from ultralyitics native Profile func
