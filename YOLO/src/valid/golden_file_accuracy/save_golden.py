import sys, os
import argparse #cli arguments
from pathlib import Path
import torch

#PATH LOGIC
HARNESS_DIR = Path(__file__).parent.parent.parent / "compiled_benchmark"
sys.path.insert(0, str(HARNESS_DIR)) #lets this file import main.py and config.py from the harness folder
os.chdir(HARNESS_DIR) #run from the harness folder so relative paths like the yolov8n.pt weights still work

import config
from main import main


#NOTE: takes two arguments, 
#      first the tag for the resulting golden file's name
#      and second the format which the data type the harness will be run in (fp32, fp16, bf16)
#      usage e.g. python save_golde.py jetson

parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 60)) #keeps help text on the same line as each flag
parser.add_argument("--name", default = "jetson", help = "platform tag for the golden file's name (defaulted to jetson)")
parser.add_argument("--precision", choices = ["fp32", "fp16", "bf16"], default = "fp32", help = "determines data type the harness runs in (defaulted to fp32, the reference)")
args = parser.parse_args()
dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16} # will be run in either (fp32, fp16, bf16)

if args.precision not in dtypes:
    sys.exit("ERROR: PRECISION MUST BE EITHER fp32, fp16, or bf16")
#runs harness once and saves what it outputs to be compared later

#temporarily override the config.py file's data type for this run only
config.DTYPE = dtypes[args.precision]

#stops GPU from quietly using a lower T32 precision for small operations
if args.precision == "fp32" and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False #disallow T32 matmuls
    torch.backends.cudnn.allow_tf32 = False       #disallow T32 convolutions


results = main("profile", config.IMG_DIR, 1)

#ensure directory for golden to be saved to exists
out_dir = config.BENCH_DIR / "golden_files"
out_dir.mkdir(parents = True, exist_ok = True)

torch.save([r[0].float().cpu() for r in results], out_dir / f"golden_{args.name}_{args.precision}.pt")
print("saved golden file at", out_dir / f"golden_{args.name}_{args.precision}.pt") #indicates where the golden file was saved, and what name it's saved under
