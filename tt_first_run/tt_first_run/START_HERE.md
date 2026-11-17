# First run on the Tenstorrent host
Open this directory inside the lab's working TT-XLA environment with the card exposed. See README.md for container setup. Use the same shell for each command.

    python -c "import torch_xla, torch_xla.runtime as xr; xr.set_device_type('TT'); print(xr.device_type(), torch_xla.device())"
    python -m pip install -r requirements.txt
    python download_model.py
    tt-smi -s
    python benchmark_latency.py --model models/tinyllama-chat --text-only --max-new-tokens 2
    bash run_min.sh

The first command should show TT and an XLA device. The download needs internet once. The downloader records the model commit in models/tinyllama-chat/download_provenance.json. Supply --revision COMMIT to match your Jetson. The two-token run checks the model path and device execution before collecting data. The mini suite compares three short runs per mode after one warm-up. Inspect card.telemetry.csv for source tt-smi -s and status ok. Short runs can finish between samples.

## Design choices
- download_model.py prepares weights and tokenizer before timing. Local files let repeated runs use the same checkpoint.
- tinyllama_stage.py owns tokenization, model loading and generation. pipeline.py routes image requests through team adapters. contracts.py defines the values passed between stages.
- benchmark_latency.py synchronizes device work around the request and records token arrival times so the host clock includes accelerator execution.
- telemetry.py samples Linux and card sensors in background threads. benchmark_cpu.py and benchmark_gpu.py enable monitoring on the same inference path.
- run_min.sh runs each mode sequentially with identical inputs so benchmark jobs do not compete for the card. Each invocation writes a new results directory.

For image runs, provide a real image, the team's importable adapter factory, its trained weights and matching alignment manifest. Then use the same suite:

    bash run_min.sh --image scene.jpg --adapters team_adapters:make_components --alignment alignment.json

See README.md for the adapter contract and CSV metric definitions.
