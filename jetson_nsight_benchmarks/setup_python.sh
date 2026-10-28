#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Activate your WORKING Jetson PyTorch environment before calling this script.
python -c 'import torch; assert torch.cuda.is_available(), "Activate an environment with CUDA-enabled Jetson PyTorch first"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pip install -r requirements.txt
python check_jetson.py
