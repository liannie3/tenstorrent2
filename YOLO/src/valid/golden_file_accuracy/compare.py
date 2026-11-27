import sys
import argparse #cli arguments
from pathlib import Path
import torch
from torchvision.ops import box_iou

#NOTE: compares a test golden file's detections against a reference golden file, image by image
#      images are matched by FILE NAME, so a --step subset (e.g. golden_jetson_bf16_step4) can be checked against
#      the full set reference (e.g. golden_jetson_fp32): every subset image is one of the reference's images
#      old golden files (plain lists, no names) are matched by position and must have the same number of images
#USAGE: python compare.py jetson_fp32 jetson_bf16_step4      (file tags, the part after "golden_")
parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("golden", help = "reference golden file tag, e.g. jetson_fp32 for golden_jetson_fp32.pt")
parser.add_argument("test", help = "golden file tag being checked, e.g. tt_bf16 or jetson_bf16_step4")
parser.add_argument("--accuracy", type = float, default = 95, help = "%% of reference boxes that must be found in the test (defaulted to 95)")
parser.add_argument("--max_conf_diff", type = float, default = 0.01, help = "max average confidence difference on matched boxes (defaulted to 0.01)")
args = parser.parse_args()

#pass file tags not paths for ease of use
GOLDEN_DIR = Path(__file__).parent.parent.parent.parent / "benchmarking" / "golden_files" #compare.py is currently in YOLO/src/valid/golden_file_accuracy, 4 parents up is YOLO folder

#helper function that loads a golden file as (names or None, list of detection tensors)
def load(tag):
    data = torch.load(GOLDEN_DIR / f"golden_{tag}.pt")
    if isinstance(data, dict): #new format, keyed by image name
        return data["names"], data["dets"]
    return None, data #old format, plain list in image order

golden_names, golden_dets = load(args.golden) #golden reference detections (e.g. golden_jetson_fp32.pt)
test_names, test_dets = load(args.test) #detections being checked (e.g. golden_tt_bf16.pt)

#PAIR UP IMAGES: by file name when both files have names, otherwise by position
if golden_names is not None and test_names is not None:
    golden_by_name = dict(zip(golden_names, golden_dets))
    missing = [n for n in test_names if n not in golden_by_name]
    if missing:
        sys.exit(f"ERROR: {len(missing)} TEST IMAGES AREN'T IN THE REFERENCE (e.g. {missing[0]}), the reference should be the full set (step 1)")
    pairs = [(golden_by_name[n], t) for n, t in zip(test_names, test_dets)]
else:
    if len(golden_dets) != len(test_dets):
        sys.exit("ERROR: FILES HAVE A DIFFERENT NUMBER OF IMAGES (old format files are matched by position, save them again to match by name)")
    pairs = list(zip(golden_dets, test_dets))
print(f"comparing {len(pairs)} images" + (f" (test is a subset of the {len(golden_dets)} reference images)" if len(pairs) < len(golden_dets) else ""))

total_golden = 0
total_test = 0
found = 0
conf_diff_sum = 0 #sum of |golden conf - test conf| over all matched boxes
conf_diff_max = 0 #largest single confidence difference
missed_borderline = 0 #measures missed reference boxes that are close to the 0.25 confidence threshold

for g, t in pairs: #same image on both sides
    #each row of a detection tensor: x1, y1, x2, y2, confidence, class
    total_golden += len(g)
    total_test += len(t)
    if len(g) == 0 or len(t) == 0:
        continue

    iou = box_iou(g[:, :4], t[:, :4]) #overlap between every golden reference box and every test box
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
            if conf_diff > conf_diff_max: #greedily check for new greatest confidence discrepancy
                conf_diff_max = conf_diff
        if best_b < 0 and g[a, 4] < 0.3: #missed, and was borderline in the reference
            missed_borderline += 1

if total_golden == 0:
    sys.exit("ERROR: REFERENCE HAS NO BOXES ON THESE IMAGES, NOTHING TO COMPARE")
accuracy = round(100 * found / total_golden, 1)
avg_conf_diff = round(conf_diff_sum / found, 5) if found else float("inf")
print("reference boxes:", total_golden, "  test boxes:", total_test)
print("reference boxes also found in test:", found, "/", total_golden, "=", accuracy, "%")
if found > 0:
    print("confidence difference on matched boxes: average", avg_conf_diff, "  max", round(conf_diff_max, 5))
#NOTE: want to shoot for below 0.01, or better, difference in average confidence scores
if missed_borderline > 0:
    print("Total missed reference boxes:", total_golden - found, "\nTotal missed borderline boxes:", missed_borderline)

#VALIDATION CHECK
if accuracy >= args.accuracy:
    print(f"    ===Golden file accuracy validation check PASSED!===\n    ===Expected {args.accuracy}% accuracy or above, actual: {accuracy}%===")
else:
    print(f"    ===Golden file accuracy validation check FAILED!=== \n    ===Expected {args.accuracy}% accuracy or above, actual: {accuracy}%===")

if avg_conf_diff <= args.max_conf_diff:
    print(f"    ===Golden file average confidence difference on matched boxes check PASSED!===\n    ===Expected confidence difference below {args.max_conf_diff}, actual: {avg_conf_diff}===")
else:
    print(f"    ===Golden file average confidence difference on matched boxes check FAILED!===\n    ===Expected confidence difference below {args.max_conf_diff}, actual: {avg_conf_diff}===")

#SUBSET REPRESENTATIVENESS (info only): does the subset look like the full reference set?
if len(pairs) < len(golden_dets):
    full = torch.cat([d for d in golden_dets if len(d)]) if any(len(d) for d in golden_dets) else torch.zeros(0, 6)
    sub = torch.cat([g for g, _ in pairs if len(g)]) if any(len(g) for g, _ in pairs) else torch.zeros(0, 6)
    print(f"\n  subset representativeness (reference detections, subset images vs all {len(golden_dets)} images)")
    print(f"  detections per image: subset {len(sub) / len(pairs):.2f}   full set {len(full) / len(golden_dets):.2f}")
    print(f"  {'class':>6} {'subset %':>9} {'full %':>8}")
    for c in sorted(set(full[:, 5].int().tolist()), key = lambda c: -(full[:, 5] == c).sum().item())[:8]: #8 most common classes
        print(f"  {c:>6} {100 * (sub[:, 5] == c).sum().item() / max(len(sub), 1):>9.1f} {100 * (full[:, 5] == c).sum().item() / len(full):>8.1f}")
