import torch
import os #allows for looping through the entire directory of test images

from ultralytics import YOLO


def main():
	model = YOLO("yolov8n.pt")

	path = "test_images/"
	#best practice to robustly define the device
	#so it only attempts to utalize cuda if a cuda gpu is present
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	for filename in os.listdir(path):
		curr_path = os.path.join(path, filename)
		testResults1 = model(curr_path) #passes the filename (path) to the model

		for result in testResults1:
  			boxes = result.boxes
  			masks = result.masks
  			keypoints = result.keypoints
  			probs = result.probs
  			print(boxes)
  			print(masks)
  			print(keypoints)
  			print(probs)
  			result.save(filename = "result_images/" + filename)

if __name__ == "__main__":
	main()
