import torch
import sys
import os

from ultralytics import YOLO

def main():
    front_view_path = ""
    prompt = ""
    if len(sys.argv) > 2:
        front_view_path = sys.argv[1] #takes image file path as arg1
        prompt = sys.argv[2] #takes user prompt as arg2
    else:
        print("Please pass arguments of file image path, and VQA prompt")

    if not front_view_path.lower().endswith((".png",".jpg",".jpeg")): #ensure valid img type
        return
    model = YOLO("yolov8n.pt") #set model as pretrained nano yolo
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #safely define device


    results = model(front_view_path, profile = True) #run detection on input img with pipeline level benchmarking

    #results[0].show() #show first, and only, res
    results[0].save(filename = "output_images/output.jpg") #save to be viewed outside of shell


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
    print(hybrid_prompt)

if __name__ == "__main__":
    main()
