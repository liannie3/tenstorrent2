"""Run the same TinyLlama request under independently selectable measurements."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from report import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["latency", "cpu", "device", "power", "all"])
    p.add_argument("--model", default="models/tinyllama-chat")
    p.add_argument("--device", type=int, default=0, help="TT runtime device ID")
    p.add_argument("--bdf", help="Full PCI address of that device from tt-smi")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--question", default="Explain the difference between a car and a pedestrian.")
    p.add_argument("--results", default="results")
    p.add_argument("--image")
    p.add_argument("--alignment")
    p.add_argument("--adapters", help="Team component factory, module:function")
    p.add_argument("--rag-query-mode", choices=["tinyllama_text","clip"], default="tinyllama_text")
    p.add_argument("--embedding-id")
    args = p.parse_args()
    if ((args.runs is not None and args.runs < 1) or (args.warmup is not None and args.warmup < 0)
            or args.max_new_tokens < 1 or args.interval_ms < 1 or not args.question.strip()):
        p.error("Check run counts, token budget, sensor interval and question.")
    visual = (args.image, args.alignment, args.adapters)
    if any(visual) and (not all(visual) or args.adapters.count(":") != 1):
        p.error("Supply --image, --alignment and --adapters module:factory together.")
    if not (Path(args.model)/"config.json").is_file():
        p.error("Model config missing. Run download_model.py or pass --model /path/to/checkpoint.")
    if "TT_METAL_HOME" not in os.environ:
        p.error("Set TT_METAL_HOME and activate its python_env first. See README.")
    base = Path(args.results).resolve()
    base.mkdir(parents=True, exist_ok=True)
    suite = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=base))
    package = Path(__file__).resolve().parent
    for mode in (["latency","cpu","device","power"] if args.mode == "all" else [args.mode]):
        output = suite / mode
        output.mkdir()
        cfg = vars(args).copy()
        for key in ("model","image","alignment"):
            if cfg[key]:
                cfg[key] = str(Path(cfg[key]).resolve())
        cfg.update(mode=mode, output=str(output), runs=args.runs if args.runs is not None else (1 if mode=="device" else 3),
                   warmup=args.warmup if args.warmup is not None else (0 if mode=="device" else 1))
        config = output/"config.json"
        config.write_text(json.dumps(cfg, indent=2))
        env = dict(os.environ, P150_CONFIG=str(config),
                   PYTHONPATH=os.pathsep.join([str(package), str(Path.cwd()), os.environ.get("PYTHONPATH","")]))
        cmd = [sys.executable]
        if mode == "device":
            env["TT_METAL_PROFILER_DIR"] = str(output/"capture")
            cmd += ["-m","tracy","-r","-p","--no-web-server","--dump-device-data-mid-run","--device-memory-profiler","-m","worker"]
        else:
            env.pop("TT_METAL_DEVICE_PROFILER", None)
            cmd += ["-m","worker"]
        print(f"Running {mode} in {output}", flush=True)
        subprocess.run(cmd, env=env, cwd=output, check=True)
        if not (output/"requests.csv").is_file():
            raise RuntimeError(f"Worker did not finish. Inspect {output}")
        if mode == "device":
            summarize(output/"capture", output/"device_ops.csv", cfg["runs"])
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
