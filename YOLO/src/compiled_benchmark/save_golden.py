import sys, torch, config
from main import main

#NOTE: takes two arguments, 
#      first the tag for the resulting golden file's name
#      and second the format which the data type the harness will be run in (fp32, fp16, bf16)
#      usage e.g. python save_golde.py jetson_fp32

if len(sys.argv) !=3:
    sys.exit("USAGE: python save_golden.py <name> <fp32|fp16|bf16>")

name = sys.argv[1]
data_type = sys.argv[2]
dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16} # will be run in either (fp32, fp16, bf16)
if data_type not in dtypes:
    sys.exit("ERROR: PRECISION MUST BE EITHER fp32, fp16, or bf16")
#runs harness once and saves what it outputs to be compared later

#temporarily override the config.py file's data type for this run only
config.DTYPE = dtypes[data_type]

#stops GPU from quietly using a lower T32 precision for small operations
if data_type == "fp32" and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False #disallow T32 matmuls
    torch.backends.cudnn.allow_tf32 = False       #disallow T32 convolutions


results = main("profile", config.IMG_DIR, 1)

#ensure directory for golden to be saved to exists
out_dir = config.BENCH_DIR / "golden_files"
out_dir.mkdir(parents = True, exist_ok = True)

torch.save([r[0].float().cpu() for r in results], out_dir / f"golden_{sys.argv[1]}.pt")
print("saved", out_dir / f"golden_{name}_{data_type}.pt") #identify where the golden file is saved and what it was called
