import sys, torch, config
from main import main

#NOTE: takes one argument, the tag for the resulting golden file's name
#      e.g. python save_golde.py jetson_fp32

#runs harness once and saves what it outputs to be compared later

results = main("profile", config.IMG_DIR, 1)

#ensure directory for golden to be saved to exists
out_dir = config.BENCH_DIR / "golden_files"
out_dir.mkdir(parents = True, exist_ok = True)

torch.save([r[0].float().cpu() for r in results], out_dir / "golden_files" , f"golden_{sys.argv[1]}.pt")
