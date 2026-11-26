import sys
import torch
from torchvision.ops import box_iou

import config

#NOTE: TAKES CLI ARGUMENTS OF GOLDEN FILE AND FILE TO BE COMPARED AGAINST GOLDEN
#      e.g. "python compare.py golden_jetson_fp32 golden_jetson_fp32"

#pass file names not paths for ease of use
golden = torch.load(config.BENCH_DIR / "golden_files" / f"golden_{sys.argv[1]}.pt")  #golden reference detections (e.g. golden_jetson_fp32.pt)
test = torch.load(config.BENCH_DIR / "golden_files" / f"golden_{sys.argv[2]}.pt") #detections being checked (e.g. golden_tt_bf16.pt)

if len(golden) != len(test):
    sys.exit("ERROR: FILES HAVE A DIFFERENT NUMBER OF IMAGES")

total_golden = 0
total_test = 0
found = 0
conf_diff_sum = 0 #sum of |golden conf - test conf| over all matched boxes
conf_diff_max = 0 #largest single confidence interval difference
missed_borderline = 0 #measures missed reference boxes that are close to the 0.25 confidence threshold

for i in range(len(golden)): #same images, same order on both systems
    #each row of a detection tensor: x1, y1, x2, y2, confidence, class
    g = golden[i] #golden tensor row
    t = test[i] #test tensor row
    total_golden += len(g)
    total_test += len(t)
    if len(g) == 0 or len(t) == 0:
        continue

    iou = box_iou(g[:, :4], t[:, :4]) #overlap between every golden box and every test box

    #overlap between every golden reference box and every test box
    for a in range(len(g)): #for each golden box
        best_b = -1
        best_iou = 0.5 #must overlap at least 50%
        for b in range(len(t)): #find the test box of the same class that overlaps it the most
            if g[a, 5] == t[b, 5] and iou[a, b].item() >= best_iou:  #same class and at least current best overlap
                best_b = b
                best_iou = iou[a, b].item()
        if best_b >= 0:
            found += 1
            conf_diff = abs(g[a, 4] - t[best_b, 4]).item() #column 4 is the confidence score
            conf_diff_sum += conf_diff
            if conf_diff > conf_diff_max: #greedily check for new greatest cnfidence discrepancy
                conf_diff_max = conf_diff
        if best_b < 0 and g[a, 4] < 0.3: #missed, and was borderline in the reference
            missed_borderline+=1

print("reference boxes:", total_golden, "  test boxes:", total_test)
if total_golden > 0:
    print("reference boxes also found in test:", found, "/", total_golden, "=", round(100 * found / total_golden, 1), "%")
if found > 0:
    print("confidence difference on matched boxes: average", round(conf_diff_sum / found, 5), "  max", round(conf_diff_max, 5))
#NOTE: want to shoot for below 0.01, or better, difference in average confidence scores
if missed_borderline > 0:
    print("Total missed reference boxes:", total_golden - found, "\nTotal missed borderline boxes:", missed_borderline)
