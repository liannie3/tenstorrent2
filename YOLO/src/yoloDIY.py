import cv2 #open source computer vision libray utilized by ultralytics
import torch
from ultralytics.nn.tasks import attempt_load_weights #ultralyitics neural network function


from ultralytics.data.augment import LetterBox #ultralytics preprocess function to resize and pad
                                               #input images to a specified shape while preserving aspect ratio
                                               #additionally updates labels & bounding boxes respectively


from ultralytics.utils.ops import non_max_suppression, scale_boxes


#                   Passes a list of numpy tensors as im
#                   
def preprocess_stage(im: list[np.ndarray], device = 'cuda:0', fp16 = False):

    #PRE-PROCESS STEP A: pre-transform input img before inference

def inference_stage():


def postprocess_stage():


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #setup phase separate of pre process

    dummy_tensor = preprocess_stage(dummy_imgs, device = device)

if __init__ == "__main__":
    main()


# 1. SETUP (From ultralytics/models/yolo/model.py & engine/model.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load_weights("yolov8n.pt", device=device)  # Loads raw PyTorch nn.Module
model.eval()

# 2. SOURCE LOADING (From ultralytics/engine/predictor.py -> load_inference_source)
orig_img = cv2.imread("image.jpg")  # Simulating raw source input (HWC, BGR)


# PRE-PROCESSING STEP
# FOLLOWING preprocessing func from ultralytics/models/yolo/detect/predict.py 

# PREPROCESS STEP A: Pre-transform input image before before inference
# Utilizing Letterbox func from (From ultralytics/data/augment.py)
# recreating pre_transform wrapper func evoked by preprocessing
img_resized = LetterBox(new_shape=(640, 640), auto=True, stride=32)(image=orig_img)

# Step B: Tensor formatting and normalization (From ultralytics/models/yolo/detect/predict.py)
im = img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
im = torch.from_numpy(im.copy()).to(device)
im = im.float() / 255.0                      # Normalize pixel values to [0.0, 1.0]
im = im[None]                                # Add batch dimension (BCHW)

# 4. INFERENCE (From ultralytics/engine/predictor.py -> inference)
with torch.no_grad():
    raw_preds = model(im) #pass to neural network

# 5. POST-PROCESSING (From ultralytics/models/yolo/detect/predict.py -> postprocess)
# Step A: Non-Maximum Suppression filtering (From ultralytics/utils/ops.py)
filtered_preds = non_max_suppression(raw_preds, conf_thres=0.25, iou_thres=0.45)

# Step B: Coordinate Rescaling (From ultralytics/utils/ops.py)
for pred in filtered_preds:
    if len(pred):
        # Maps coordinates back from 640x640 space to the original image shape
        pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], orig_img.shape).round()

# 6. RESULTS
print("Processed Bounding Boxes [x1, y1, x2, y2, confidence, class]:")
print(filtered_preds)
