from ultralytics import YOLO
import cv2, glob, numpy as np
imgs = [cv2.imread(p) for p in sorted(glob.glob("input_images/CAM_FRONT/*.jpg"))]
m = YOLO("yolov8n.pt")
res = [m.predict(im, verbose=False)[0] for im in imgs]
print({k: round(np.mean([r.speed[k] for r in res[1:]]), 2) for k in ("preprocess", "inference", "postprocess")})
