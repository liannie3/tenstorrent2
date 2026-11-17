"""The common worker for all four mini benchmarks. Config comes from bench.py."""
import importlib
import json
import os
import platform
import resource
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

from metrics import Observer, Sampler, energy, sensors, write_csv


def run(config, device, tt):
    import torch
    from tinyllama_stage import TinyLlamaStage
    root = Path(config["output"])
    mode = config["mode"]
    llm = TinyLlamaStage(config["model"], device=device, alignment_manifest=config["alignment"])
    pipeline = None
    if config["adapters"]:
        from pipeline import AVPipeline
        module, factory = config["adapters"].split(":")
        pipeline = AVPipeline(llm, **getattr(importlib.import_module(module), factory)(llm),
                              rag_query_mode=config["rag_query_mode"], embedding_id=config["embedding_id"])
    def request(observer=None):
        args = dict(max_new_tokens=config["max_new_tokens"], benchmark=observer)
        if pipeline:
            return pipeline.run(config["image"], config["question"], **args).answer
        return llm.answer(config["question"], **args)

    for _ in range(config["warmup"]):
        request()
    tt.synchronize_device(device)
    paths = sensors(config["bdf"]) if mode == "power" else {}
    if paths:
        (root/"sensors.json").write_text(json.dumps({k:str(v[0]) for k,v in paths.items()}, indent=2))
    mark = None
    if mode == "device":
        from tracy import signpost as mark
    results, stages, tokens, samples, answers = [], [], [], [], []
    interval = config["interval_ms"]/1000
    for i in range(1, config["runs"]+1):
        obs = Observer(mark)
        if mark:
            mark(header="measured_begin", message=str(i))
        sampler = Sampler(paths, interval) if paths else nullcontext()
        with sampler as sensor:
            cpu_start = time.process_time()
            start = time.perf_counter()
            answer = request(obs)
            tt.synchronize_device(device)
            end = time.perf_counter()
            cpu_used = time.process_time()-cpu_start
        if mark:
            mark(header="measured_end", message=str(i))
        row = dict(request=i, **obs.timing(start, end), finish_reason=answer.finish_reason)
        if mode == "cpu":
            row.update(process_cpu_ms=cpu_used*1000, process_cpu_pct=100*cpu_used/(end-start),
                       process_lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if paths:
            joules, coverage = energy(sensor.rows, start, end, interval)
            row.update(board_energy_j=joules, power_coverage=coverage,
                       board_mean_w=joules/(end-start) if joules is not None else None,
                       j_per_token=joules/len(obs.tokens) if joules is not None else None,
                       tokens_per_j=len(obs.tokens)/joules if joules else None)
            for key in ("temp_c", "aiclk_mhz"):
                vals = [r[key] for r in sensor.rows if start <= r["time_s"] <= end and r.get(key) is not None]
                row["mean_"+key] = sum(vals)/len(vals) if vals else None
                row["max_"+key] = max(vals, default=None)
            for s in sensor.rows:
                samples.append(dict(request=i, relative_s=s["time_s"]-start,
                                    **{k:v for k,v in s.items() if k != "time_s"}))
        results.append(row)
        stages.extend(dict(request=i, **s) for s in obs.spans)
        tokens.extend(dict(request=i, token_index=j, token_id=t, ready_ms=(stamp-start)*1000)
                      for j,(stamp,t) in enumerate(zip(obs.tokens, answer.token_ids),1))
        answers.append(dict(request=i, text=answer.text, ids=answer.token_ids))
    for name, rows in [("requests",results),("stages",stages),("tokens",tokens),("sensors",samples)]:
        write_csv(root/(name+".csv"), rows)
    (root/"answers.json").write_text(json.dumps(answers, indent=2))
    print(json.dumps(results, indent=2))


def main():
    import ttnn as tt
    import torch
    import transformers
    cfg = json.loads(Path(os.environ["P150_CONFIG"]).read_text())
    root = Path(cfg["output"])
    git = subprocess.run(["git","-C",os.environ["TT_METAL_HOME"],"rev-parse","HEAD"],
                         capture_output=True,text=True)
    provenance = Path(cfg["model"])/"download_provenance.json"
    (root/"environment.json").write_text(json.dumps(dict(kernel=platform.release(), python=platform.python_version(),
        torch=str(torch.__version__),transformers=transformers.__version__,ttnn_path=tt.__file__,
        metal_commit=git.stdout.strip(), model_revision=json.loads(provenance.read_text()) if provenance.exists() else None,
        dtype="bfloat16",device_id=cfg["device"]),indent=2))
    d = tt.open_device(device_id=cfg["device"])
    try:
        with torch.inference_mode():
            run(cfg, d, tt)
    finally:
        tt.close_device(d)


if __name__ == "__main__":
    main()
