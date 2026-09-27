import torch
import sys
import os

from ultralytics import YOLO

def displayResults(testResults1, filename):
        for result in testResults1:
                boxes = result.boxes
                masks = result.masks
                probs = result.probs
                keypoints = result.keypoints
                print(boxes)
                print(masks)
                print(probs)
                print(keypoints)
                result.save(filename = "result_images/" + filename)

def main():
        setting = ""
        if len(sys.argv) > 1:
                setting = sys.argv[1]
        model = YOLO("yolov8n.pt")

        path = "test_images/"
        #best practice to robustly define the device
        #so it only attempts to utalize cuda if a cuda gpu is present
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for filename in os.listdir(path):
                if(setting == "complete"):
                        if not filename.lower().endswith((".png", ".jpeg", ".jpg")):
                                continue #skips any non applicable test files
                        full_path = os.path.join(path, filename)
                        testResults1 = model(full_path) #passes the filename (path) to the model
                        displayResults(testResults1, filename)

                elif(setting.lower().endswith((".png", ".jpeg", ".jpg"))):
                        full_path = os.path.join(path, setting)
                        testResults1 = model(full_path)
                        displayResults(testResults1, setting)
                        break
                else:
                        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                                break
                        full_path = os.path.join(path, filename)
                        testResults1 = model(full_path)
                        displayResults(testResults1, filename)
                        break;
if __name__ == "__main__":
        main()
