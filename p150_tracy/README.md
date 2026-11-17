# P150 Blackhole • TinyLlama and Tracy

Start here on the Linux computer that physically hosts the P150. The first milestone is a correct token from the card. Then collect latency, CPU, Tensix and board-energy measurements as separate experiments.

## Before running any mini benchmark

1. **Identify the machine and board.** The pinned TT-Metal installation guide lists Ubuntu 22.04 and Python 3.10 for Blackhole. Follow the P150 variant's hardware installation guide for power, cooling and slot requirements. On the host, run `lspci -nn -d 1e52:`. No matching device at this point means Linux has not enumerated the board. Resolve that before installing Python libraries. On a shared machine, check the existing installation and arrange an idle card before any changes or measurements.
2. **Install the host stack once.** On a newly installed host, an administrator needs TT-KMD, matching Blackhole firmware, TT-SMI, kernel headers, 1 GiB HugePages and user access to the device. The pinned source guide links the installer below. Review its proposed versions against the Blackhole row in the guide and the host's existing firmware. A firmware update resets devices. Reboot after initial driver/HugePages setup if directed. Skip this installation on a functioning managed host. The benchmark scripts run as your normal user.

   ```bash
   sudo apt update
   sudo apt install git curl pciutils unzip build-essential "linux-headers-$(uname -r)"
   mkdir -p "$HOME/p150-host-setup"
   cd "$HOME/p150-host-setup"
   curl -fLO https://github.com/tenstorrent/tt-installer/releases/download/v2.1.0/install.sh
   less install.sh
   bash install.sh --install-container-runtime=no
   ```

3. **Verify the host after reboot.** Run the following from a normal shell. The board column should identify a P150 variant. Record its device ID, PCI bus address, driver and firmware versions. If your installed TT-SMI is interactive, quit after noting them. Confirm the device node is accessible and a hugetlbfs mount for 1 GiB pages exists. A zero free count before opening the device can indicate that other jobs already hold the pages.

   ```bash
   tt-smi
   ls -l /dev/tenstorrent/
   findmnt -t hugetlbfs -o TARGET,OPTIONS
   cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages
   ```

4. **Unzip the kit and build TT-Metal with Tracy.** From the directory containing this ZIP, run the following. The two calls to the helper first fetch a pinned upstream checkout and then build that same checkout after its system dependencies are available. The `sudo` dependency step changes host packages. On a managed host the administrator can do that step or supply a matching profiler-enabled checkout. Keep the build in a user-owned directory with free space for source, build artifacts, Python packages, model and traces. The build and model download require network access.

   ```bash
   unzip p150_tracy.zip
   cd p150_tracy
   export KIT="$PWD"
   export TT_METAL_HOME="$HOME/p150-tt-metal"
   bash setup_source.sh "$TT_METAL_HOME" --checkout-only
   (cd "$TT_METAL_HOME" && sudo ./install_dependencies.sh)
   bash setup_source.sh "$TT_METAL_HOME"
   source "$TT_METAL_HOME/python_env/bin/activate"
   export PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
   cd "$KIT"
   bash preflight.sh --device 0
   ```

   The preflight must print a passing device matrix multiply. A successful Python import alone does not test the hardware. The source environment installs TT-NN, its matching Tracy Python module and capture tools, host PyTorch, Transformers, the tokenizer and Hugging Face client. The pinned upstream requirements select Transformers 5.12.1. Leave the Jetson requirements in the Jetson environment. This path uses neither Docker nor torch-xla. On your next login, set `TT_METAL_HOME`, activate `python_env`, export `PYTHONPATH` and return to the kit directory.

5. **Prepare the actual checkpoint.** Use the same TinyLlama revision you benchmarked on Jetson when comparing boards. If copying that checkpoint, point `--model` at its directory in every command. Otherwise, from the kit directory run:

   ```bash
   python download_model.py --destination models/tinyllama-chat
   python check.py --device 0 --model models/tinyllama-chat
   ```

   The downloader records the resolved Hugging Face SHA. `--revision SHA` can reproduce a known Jetson revision. The check compares P150 logits against a CPU FP32 run for a short prompt and two teacher-forced decode steps. Acceptance requires finite cosine similarity of at least 0.99 and the reference top token in the P150 top five on each step. It prints the actual scores and IDs. A failure here means timings are premature. An untimed model run requires sufficient host RAM for the reference model and card memory for BF16 weights.

6. **Verify a short request and a short Tracy capture.** Start with two output tokens. Open the printed results folder. Inspect `answers.json`, the token IDs and finish reason. The profiler smoke test must produce a nonempty capture, a vendor op CSV with device timings and the filtered `device_ops.csv`.

   ```bash
   python bench.py latency --runs 1 --warmup 0 --max-new-tokens 2
   python bench.py device --runs 1 --warmup 0 --max-new-tokens 2
   ```

## Run the mini benchmarks separately

From the activated environment and kit directory:

```bash
python bench.py latency --warmup 2 --runs 10 --max-new-tokens 32
python bench.py cpu     --warmup 2 --runs 10 --max-new-tokens 32
python bench.py device  --warmup 1 --runs 1  --max-new-tokens 4
python bench.py power   --warmup 2 --runs 10 --max-new-tokens 32 --interval-ms 100
```

Replace `--device 0` when the assigned TT runtime device differs. For power sampling with multiple cards, also pass the **matching** full PCI address via `--bdf`, as displayed by TT-SMI. Thermal/power results use that PCI function. For example, if your assigned card is device 1 at bus 0000:41:00.0, append `--device 1 --bdf 0000:41:00.0`. Record the mapping when using any device visibility override. To run a common workload sequentially in four fresh processes:

```bash
python bench.py all --warmup 1 --runs 2 --max-new-tokens 4
```

Use `--question "Describe what a pedestrian is."` to match an earlier demo. Each invocation creates a new results directory. The device default favors a small first trace (one measured request, no warmup). Use explicit warmups for steady-state profiling. Short budget, batch size one and early EOS are intentional. For comparable performance, hold prompt, checkpoint revision, workload, warmup count and token budget fixed, and report the actual token counts. Use latency mode for headline numbers because Tracy capture adds overhead. Each reported request includes prompt handling, prefill, decode, final text conversion and final synchronization. Model construction and warmups are outside the request timers.

## Metrics and files

| Experiment | Where to look | How to use it |
|---|---|---|
| Latency | `requests.csv`, `tokens.csv` | TTFT measures request start to first token on the host. TPOT is (last − first token time)/(generated tokens − 1). Decode tokens/s is its reciprocal in seconds. Request tokens/s includes startup of that request and final text decoding. TPOT and decode rate are blank for one-token answers. |
| Stage costs | `stages.csv` | Compare host wall time for prefill and each cached decode. The last-position logits readback waits for device output. Token selection runs on the host. Parent stages can contain child stages, so summing every row double-counts. |
| Host CPU | CPU mode, `requests.csv` | Process CPU time divided by elapsed wall time, where 100% means one fully occupied logical CPU. Peak RSS is the process lifetime high-water mark and includes model loading. |
| Tensix device | `device_ops.csv`, `capture/**/ops_perf_results*.csv`, `*.tracy` | Start with ops ranked by summed hardware kernel ms during measured requests. Then inspect individual device, core count, host duration, firmware duration, kernel duration and per-core fields where populated. Their time windows can overlap. The sum is for ranking operations. Use the trace for the critical path. Device capture also requests allocator profiling for L1/DRAM. |
| Board telemetry | Power mode, `sensors.csv`, `sensors.json`, `requests.csv` | TT-KMD readings include power (W), ASIC temperature (°C) and AI clock (MHz) when exposed. Read timestamps are relative to request start. Negative and post-request rows bracket the window. Inspect sample coverage, mean power, J/request, J/token and tokens/J. |
| Reproduction | `config.json`, `environment.json`, `answers.json` | Contains settings, prompt, versions, source commit, checkpoint provenance when present, generated token IDs, finish reason in the request CSV, and decoded text. |

To view a capture, copy the generated `.tracy` file to a workstation with the compatible Tenstorrent Tracy viewer, open the saved trace, and find `measured_begin`, `begin_prefill` and `begin_decode_step`. Alternatively, open the vendor CSV in a spreadsheet. The capture includes setup and warmup. The summary admits device rows only between measured begin/end markers. Device utilization percentage, achieved DRAM bandwidth, and whole-system energy cannot be inferred from those durations.

Power integration uses trapezoids clipped to each request, requires two in-window power readings and full boundary coverage, and discards intervals over three sample periods. Missing or sparse samples produce blank energy fields. Increase measured request length when EOS allows. Multiple short requests do not improve each other's coverage. Board sensor readings include idle board power. The card's sensor scope differs from an entire Jetson system or a wall-power meter.

## How the model maps to your source files

The host retains your chat template, text/visual input ordering, greedy selection, EOS handling, tokenizer, embedding lookup and pipeline contracts. The new `decoder.py` uploads BF16 weights once, runs RMSNorm, Q/K/V projections, rotary application, grouped-query attention, gated MLP, residuals, final norm and vocabulary projection with TT-NN, and keeps per-layer KV tensors on device between tokens. RoPE constants and the causal mask are prepared on the host. The last-position logits are read back per step. This eager reference implementation expands grouped KV heads for attention and extends the cache by concatenation. Those costs are visible in Tracy and can differ substantially from optimized vendor serving kernels. TTFT and throughput describe this implementation.

The optional visual path uses your original pipeline, adapter classes and manifest contract. Supply the team's real adapter factory, matching trained projector/checkpoint, and image; pass `--adapters my_team:make_components --alignment trained.json --image scene.jpg`. The factory takes the LLM stage. Projected tensors enter the stage as CPU BF16. The example manifest is a schema example. A text-only checkpoint by itself cannot validate the visual path.

## If the first run fails

| Symptom | Next check |
|---|---|
| PCIe entry exists, device open fails | Check TT-SMI, permissions on the assigned device node, 1 GiB HugePages after reboot and whether another workload holds the board. |
| `import tracy` or capture executable missing | Check `which python`, `TT_METAL_HOME` and preflight's module paths. Rebuild the pinned checkout with Tracy enabled. The matching CLI is `python -m tracy`. |
| Model check fails with shape/layout or poor logit match | Save the first failing stage and full traceback, runtime commit and check output. Resolve before interpreting benchmark timings. |
| Trace is empty or has no device rows | Verify the profiler-enabled build, disk space and captured worker errors. Repeat the two-token smoke test before lengthening the capture. |
| Temperature or energy is blank | Inspect `sensors.json`, `sensors.csv` and the matching board in TT-SMI. Missing driver channels or sparse samples are reported as unavailable. |
| Another user needs the host | Finish the active worker normally. Each worker closes its device. Shared driver, firmware, HugePages and checkout installation are separate from a benchmark run. |

## Validation and sources

Local host tests: `python -m unittest discover -v`. An additional development test compared the decoder algebra on a small random Hugging Face Llama model using CPU tensor equivalents: prefill and three cached steps each exceeded cosine 0.9999. No physical P150, on-card precision check, profiler capture, sensor coverage or trained team adapter was available in the development environment. The numbers from that algebra check are not hardware performance measurements.

- [Blackhole card installation](https://docs.tenstorrent.com/aibs/blackhole/installation.html)
- [Tenstorrent host setup and board identification](https://docs.tenstorrent.com/getting-started/README.html)
- [Pinned TT-Metal installation guide](https://github.com/tenstorrent/tt-metal/blob/180d922b2112b2dc17725036b3bae71852b43b6f/INSTALLING.md)
- [Pinned Tracy CLI](https://github.com/tenstorrent/tt-metal/blob/180d922b2112b2dc17725036b3bae71852b43b6f/tools/tracy/__main__.py)
- [Tracy profiler and viewer](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/tracy_profiler.html)
- [Blackhole driver telemetry channels](https://github.com/tenstorrent/tt-kmd/blob/main/blackhole.c)
