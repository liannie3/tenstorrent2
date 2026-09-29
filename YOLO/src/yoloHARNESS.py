import torch
import sys
import os
import time
import statistics
import subprocess

from ultralytics import YOLO

monitor = subprocess.Popen(
    ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu",
     "--format=csv", "-l", "1"],
    stdout=open("gpu_log.csv", "w")
)

try:
    def main():
        front_view_path = ""
        prompt = ""
        inference_times = []
        
        if len(sys.argv) > 2:
            test_database_path = sys.argv[1] #takes image file path as arg1
            prompt = sys.argv[2] #takes user prompt as arg2
        else:
            print("Please pass arguments of path to image database directory, and sample VQA prompt")

        if not os.path.isdir(test_database_path): #ensure path is actually a directory
            print("Please pass valid path to image databse directory")
            return
            
        model = YOLO("yolov8n.pt") #set model as pretrained nano yolo
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu' #safely define device

        #warmup model
        for i in 30:
            results = model("input_images/cat_dog_person.jpg", profile = False, verbose = False) #run detect on sample image

        #run model on dataset, preferably 128 images of imgsz 640
        for image_path in test_database_path:
            results = model(image_path, device = device, verbose = False)[0] #since we only pass one image at a time just use 0th index of results
            #detect classes undetected in prompt to add front-end context
            cat_list = []
            for class_tensor in results[0].boxes.cls:
                class_id = int(class_tensor) #class id are encoded  as ints
                class_name = model.names[class_id] #fetch string associated with class id
                if not ( (class_name in prompt) or (class_name in cat_list) ):
                    cat_list.append(class_name) 
            
            #format T_obj into readable english
            cat = ""
            i = 0;
            for word in cat_list:
                if i == len(cat_list)-1:
                    cat += "and " + cat_list[i]
                else:
                    cat = cat + cat_list[i] + ", "
                i+=1
                    
            
            #create T' = [T; T_obj]
            hybrid_prompt = prompt + ". Notably scan for: " + cat + " objects during analysis."
            print(hybrid_promp)

            #record time for inference
            inference_times.append(results.speed["inference"])

        #print overall stastics on inference from runs
        print(f"median: {statistics.median(times):.2f} ms")
        print(f"mean: {statistics.mean(times):.2f} ms")
        print(f"stdev: {statistics.stdev(times):.2f} ms")
        print(f"min & max: {min(times):.2f} & {max(times):.2f}")

    if __name__ == "__main__":
        main()

finally:
    #terminate subprocess
    monitor.terminate()
    monitor.wait()
