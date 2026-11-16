import sys, torch, config
from main import main

#runs harness once and saves what it outputs to be compared later

results = main("profile", config.IMG_DIR, 1)                  
torch.save([r[0].float().cpu() for r in results], f"golden_{sys.argv[1]}.pt")
