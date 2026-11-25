import torch
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	nvidia_format = "engine" #stores the export format of the benchmark, engine represents nvidia's tensorRT
	unbiased_format = "torchscript" #common baseline without specific software stack optimization
					#since we're considering architecture differences, this is the most fair
					#baseline of available options

	print("Torch Script export format"); print()
	benchmark(model = "yolov8n.pt", data = "coco8.yaml", imgsz = 640, device = device, format = unbiased_format)


if __name__ == "__main__":
        main()
