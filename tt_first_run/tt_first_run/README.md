# TinyLlama latency on a Blackhole p150a

Use the same local TinyLlama-1.1B-Chat-v1.0 checkpoint as the Jetson. Copy the checkpoint, including tokenizer files and safetensors, into models/tinyllama-chat or pass its absolute path. The code uses BF16, batch size one, greedy decoding, a fresh KV cache per request, and the Jetson prompt and stop rules.

## Environment and first run

Run on the Linux machine with the p150a. Use your lab's TT-XLA PyTorch environment. For a fresh setup, follow the [official getting-started guide](https://github.com/tenstorrent/tt-xla/blob/main/docs/source/getting_started.md). Its Docker example can be run from this extracted folder with the project mounted:

    docker run -it --rm --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G -v "$PWD:/project" -w /project ghcr.io/tenstorrent/tt-xla-slim:latest

If using this mount, put the checkpoint under the project or mount it separately. Install transformers and sentencepiece inside the environment if missing. Keep the torch and torch-xla packages paired with the TT release. Record the container digest and checkpoint revision with the experiment.

    python -c "import torch_xla, torch_xla.runtime as xr; xr.set_device_type('TT'); print(xr.device_type(), torch_xla.device())"
    python benchmark_latency.py --model models/tinyllama-chat --text-only --max-new-tokens 2
    python benchmark_latency.py --model models/tinyllama-chat --text-only --question "Describe what a pedestrian is." --max-new-tokens 64 --warmup 3 --runs 10 --output results/text_latency.csv

The first command checks the backend, the second prints an ordinary answer, and the third records timed requests. Look for two token IDs in the ordinary answer. If the first ID is EOS, use another prompt to exercise cached decoding. If compilation fails, retain the traceback, runtime versions, and whether the one-token or two-token run failed. The upstream TinyLlama demo exercises a different checkpoint and one uncached forward pass. Full cached Chat inference requires on-card verification. Changing cache shape during decoding can compile additional graphs. Repeat the exact measured workload during warm-up and inspect run-to-run variation.

## Team pipeline

    python benchmark_latency.py --model models/tinyllama-chat --image scene.jpg --question "Describe the scene." --alignment /path/to/alignment.json --adapters team_adapters:make_components --max-new-tokens 64 --warmup 3 --runs 10 --output results/scene_latency.csv

Supply a real image and your importable team factory and manifest. The factory receives the TT stage and returns yolo, vision, and optionally rag. Vision must return projected tensors on stage.device in stage.dtype. Team stages must complete or expose all dependencies by the time run returns. If they use independent asynchronous devices or queues, join their work in the adapter. Document placement of each stage when comparing totals.

## What the CSV measures

- pipeline_latency_ms covers the entire selected request, with a blocking TT fence at the end. The start follows a fence that clears previous work.
- latency_ms covers TinyLlama.answer from prompt preparation through detokenization. In image mode YOLO, vision, and retrieval precede this timer.
- ttft_ms ends at the first selected token ID available on the host, relative to the start of the LLM call.
- tpot_ms is the time from first to last host-visible token divided by the number of intervals. One-token outputs leave the cell empty.
- output_tokens_per_second divides all generated IDs, including EOS, by LLM request time. Each row retains the answer and IDs to check the work performed.
- Warm-up, model loading and CSV writing are outside the recorded intervals. TTFT includes host tokenization and transfer. TT uses BF16 and Jetson uses FP16. The Jetson Nsight observer also adds overhead. Match scope, checkpoint, prompts, output counts, and stopping reason before interpreting a difference.

## CPU, accelerator and temperature telemetry

The same pipeline now accepts optional telemetry. The new benchmark_cpu.py and benchmark_gpu.py scripts enable it by default and label the experiment. Both collect the same available hardware readings. Their names preserve the Jetson workflow. They do not change inference or enable kernel profiling. You can also add --telemetry to benchmark_latency.py.

    python benchmark_cpu.py --model models/tinyllama-chat --text-only --warmup 3 --runs 10 --max-new-tokens 64 --output results/tt_cpu.csv
    python benchmark_gpu.py --model models/tinyllama-chat --text-only --warmup 3 --runs 10 --max-new-tokens 64 --tt-smi --output results/tt_accelerator.csv

The accelerator command samples tt-smi -s in addition to Linux sensors. Confirm tt-smi -s works on the card host. Use your lab's compatible TT-SMI installation. No additional Python packages are required by this monitor. The TT-SMI subprocess must be available in the same environment/container as the benchmark. Errors such as missing executable, timeout, or device access conflicts become status records in the telemetry CSV. They do not turn into zero readings. Disable --tt-smi if your driver/runtime rejects concurrent access and use sysfs or your installed telemetry service.

Output includes the original request CSV and a second file such as results/tt_accelerator.telemetry.csv. The request CSV gains process_cpu_percent_one_core and a telemetry_summary JSON list with count/min/mean/max per source, metric and device/sensor. Means are sample arithmetic means. Longer requests can have more samples. Empty lists mean no complete samples were acquired within that request.

### Where readings come from

| Value | Source and calculation | Scope |
| --- | --- | --- |
| Process CPU % | resource.getrusage(RUSAGE_SELF) user + system CPU seconds divided by elapsed wall seconds, multiplied by 100 | Benchmark process and all its threads, including its monitoring threads. Child processes are excluded. 100% is one occupied CPU core, so totals can exceed 100%. |
| System CPU % | Deltas of /proc/stat. Busy = total minus idle and iowait. Guest counters are omitted because user/nice already include guest time. | All host CPU cores, normalized to 0–100%. |
| Process RAM | Resident pages from /proc/self/statm times the OS page size | Resident process memory, including shared resident pages. |
| Host RAM | MemTotal and MemAvailable from /proc/meminfo, converted from KiB to bytes | Whole host, potentially including other jobs. |
| CPU temperature | /sys/class/hwmon sensors with coretemp, k10temp, zenpower, cpu_thermal drivers | Available CPU package/core sensors, each retaining its label. |
| TT temperature and electrical sensors | hwmon sensors whose device or ancestor PCI vendor is 0x1e52 | Hardware sensors exposed by the Tenstorrent driver. Sensor labels and PCI paths identify the device. |
| TT ASIC temperature, clock, core/board power, voltage, current and fan RPM | Known numeric fields of tt-smi -s JSON | Snapshot values reported by TT-SMI/firmware. Core power and board power remain separate. |
| TT worker activity, queue utilization, allocated DRAM | Optional existing tt-telemetry Prometheus endpoint | Runtime-dependent device/CQ readings, preserving all topology labels. |

hwmon temp inputs are divided by 1000 to get degrees Celsius, power inputs by 1,000,000 to get watts, and current/voltage inputs by 1000 to get amps/volts. TT-SMI's already formatted fields are read in their documented units. Its current field is electrical current in amps. Device activity is obtained only from the explicit runtime utilization metrics. Every TT-SMI sample also retains the original device JSON, including firmware and raw telemetry fields, to inspect schema or sensor validity issues. Firmware/tool placeholders may still appear as numeric values from the upstream source. Validate questionable zeros against raw data and tool status.

### Optional runtime activity and device memory

If your lab already runs Tenstorrent's tt-telemetry collector, pass its actual /metrics URL. For a collector forwarded locally to port 8080, for example:

    python benchmark_gpu.py --model models/tinyllama-chat --text-only --warmup 3 --runs 10 --max-new-tokens 64 --metrics-url http://127.0.0.1:8080/metrics --output results/tt_runtime.csv

The program reads this endpoint and does not install or start a collector. In a container, use an address reachable from that container. Configure the URL for the host with your p150a. Metrics retain chip/tray/cq labels rather than assuming the snapshot's enumeration index matches an XLA ordinal. On multi-card hosts, use the PCI identity from hwmon/TT-SMI and your collector's topology mapping to select the card used by the workload.

- tt_dispatch_core_efficiency_percent measures worker-core active time.
- tt_dispatch_utilization_percent measures command-queue busy time per CQ.
- tt_device_dram_used_megabytes and tt_device_dram_total_megabytes are retained in MiB. Allocated memory is reported across attached processes, not attributed exclusively to this model.
- tt_last_update_timestamp and tt_telemetry_collection_interval_seconds are retained when exported so you can detect repeated/stale samples. The reader does not manufacture fresh hardware timestamps.

Dispatch availability depends on compatible firmware, runtime telemetry, and worker dispatch support. The documented minimum firmware for dispatch telemetry is 19.12.0 on Blackhole and 19.13.0 on Wormhole. Missing activity metrics remain absent. The memory exporter can return zero for incompatible shared-memory layouts or absent allocator state. Consult the collector's logs before interpreting those values. These measurements have different definitions from NVIDIA GPU utilization.

### Timing and interpretation

Collection starts after warm-up, continues across measured requests, and stops before file export. It uses background threads, with default host/sensor polling every 200 ms and TT-SMI/endpoint polling every 2000 ms. --interval-ms and --device-interval-ms change those requested cadences. Reads can take longer. TT-SMI subprocesses have a five-second timeout, HTTP reads a three-second timeout. A slow source can delay the next device poll. Shutdown waits for pending reads outside the latency interval.

Each telemetry CSV row contains read_start_ns/read_end_ns on the same monotonic clock as request timing, a wall-clock receipt timestamp, source, metric, value, unit, sensor/device labels and status. run is populated only if the complete acquisition interval fits inside a measured request. overlapping_runs records all overlapping requests, including boundary-crossing reads. CPU percentage intervals use the preceding counter sample as their start. Firmware sensor refresh can precede the read, so acquisition containment does not prove exclusive attribution to the request.

The sampler records all visible matching devices/sensors and preserves identity. Temperature, power and host totals include other workloads on that hardware. A short request can have no complete temperature sample. Increasing the number of requests provides a longer timeline, while making an individual request longer is more likely to produce samples entirely inside it. Use longer workloads or an appropriately shorter polling interval when needed.

For a latency baseline, run benchmark_latency.py without telemetry options. For the instrumented CPU/GPU runs, retain telemetry_enabled in the results. Polling consumes host CPU, and TT-SMI can add driver traffic. Process CPU % includes the monitor threads. System CPU % also includes the TT-SMI child and any external collector.

### Validation and references

    python -m unittest -v test_telemetry

Seven local tests cover snapshot parsing/units/multiple devices, Prometheus parsing, hwmon scaling/identity, boundary-crossing reads, tool failure, real Linux host sampling/thread cleanup and benchmark integration with a synthetic inference fixture. This environment has no p150a. No TT hardware performance, cached generation or real device telemetry was validated here. This update adds request-scoped telemetry. Internal per-layer device profiling remains a separate implementation step.

- [TT-SMI fields and snapshot interface](https://github.com/tenstorrent/tt-smi)
- [TT telemetry metric definitions and availability](https://docs.tenstorrent.com/tt-telemetry/latest/metrics.html)
- [TT telemetry endpoint](https://docs.tenstorrent.com/tt-telemetry/latest/index.html)
- [Linux hwmon interface units](https://docs.kernel.org/hwmon/sysfs-interface.html)
