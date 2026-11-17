"""Single-process application latency harness. Both modes call the same pipeline."""
import argparse
from contextlib import contextmanager, nullcontext
import csv
from dataclasses import asdict
from importlib import import_module, metadata
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter_ns


class RequestTimer:
    def __init__(self):
        self.tokens = []

    def span(self, name):
        return nullcontext()

    @contextmanager
    def request(self, stage):
        self.start = perf_counter_ns()
        yield
        self.end = perf_counter_ns()

    def input_ready(self, count):
        self.inputs = int(count)

    def token_ready(self):
        self.tokens.append(perf_counter_ns())

    def metrics(self):
        count = len(self.tokens)
        elapsed = self.end - self.start
        return dict(input_tokens=self.inputs, output_tokens=count,
                    latency_ms=elapsed / 1e6, ttft_ms=(self.tokens[0] - self.start) / 1e6,
                    tpot_ms=(self.tokens[-1] - self.tokens[0]) / ((count - 1) * 1e6) if count > 1 else None,
                    output_tokens_per_second=count * 1e9 / elapsed)


def main(mode="latency"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--telemetry', action='store_true', help='Sample CPU/RAM and hwmon sensors')
    parser.add_argument('--interval-ms', type=float, default=200)
    parser.add_argument('--device-interval-ms', type=float, default=2000)
    parser.add_argument('--tt-smi', action='store_true', help='Also sample TT-SMI JSON snapshots')
    parser.add_argument('--metrics-url', help='Optional running tt-telemetry Prometheus /metrics endpoint')
    parser.add_argument('--model', required=True, help='Local checkpoint directory')
    parser.add_argument('--question', default='Describe what a pedestrian is.')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--output', type=Path, help='CSV destination. Omit for ordinary inference.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text-only', action='store_true')
    input_group.add_argument('--image')
    parser.add_argument('--adapters', help='module:factory from the team pipeline')
    parser.add_argument('--alignment')
    parser.add_argument('--rag-query-mode', choices=['tinyllama_text','clip'], default='tinyllama_text')
    parser.add_argument('--clip-embedding-id')
    args = parser.parse_args()
    if args.image and not (args.alignment and args.adapters and ':' in args.adapters):
        parser.error('Image mode requires --alignment and --adapters module:factory')
    if args.text_only and (args.alignment or args.adapters):
        parser.error('Use --image with visual adapters')
    if min(args.max_new_tokens, args.runs) < 1 or args.warmup < 0 or not args.question.strip():
        parser.error('Check the question, token budget, warmup and run count')
    if any(not math.isfinite(x) or x < 20 for x in (args.interval_ms, args.device_interval_ms)):
        parser.error('Sampling intervals must be finite and at least 20 ms')
    telemetry_enabled = args.telemetry or args.tt_smi or args.metrics_url or mode != 'latency'
    if telemetry_enabled and args.output is None:
        parser.error('Telemetry requires --output')
    telemetry_path = args.output.with_suffix('.telemetry.csv') if args.output else None
    if args.output and (args.output.exists() or (telemetry_enabled and telemetry_path.exists())):
        parser.error('Choose a new output filename')

    from tinyllama_stage import TinyLlamaStage
    import torch_xla
    from pipeline import AVPipeline

    stage = TinyLlamaStage(args.model, alignment_manifest=args.alignment)
    if args.image:
        module, factory = args.adapters.split(':', 1)
        pipeline = AVPipeline(stage, **getattr(import_module(module), factory)(stage),
                              rag_query_mode=args.rag_query_mode, embedding_id=args.clip_embedding_id)

    def run(observer=None):
        options = dict(max_new_tokens=args.max_new_tokens, benchmark=observer)
        if args.image:
            return pipeline.run(args.image, args.question, **options)
        return stage.answer(args.question, **options)

    if args.output is None:
        print(json.dumps(asdict(run()), ensure_ascii=False))
        return

    for _ in range(args.warmup):
        run()
        torch_xla.sync(wait=True)

    rows = []
    versions = {}
    for package in ('torch', 'torch-xla', 'transformers', 'pjrt-plugin-tt'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = 'unreported'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    monitor = None
    if telemetry_enabled:
        from telemetry import TelemetryMonitor
        monitor = TelemetryMonitor(args.interval_ms / 1000, args.device_interval_ms / 1000,
                                   args.tt_smi, args.metrics_url)
        monitor.start()
    try:
        for number in range(1, args.runs + 1):
            timer = RequestTimer()
            torch_xla.sync(wait=True)
            if monitor:
                monitor.begin_request(number)
            start = perf_counter_ns()
            result = run(timer)
            torch_xla.sync(wait=True)
            end = perf_counter_ns()
            cpu_percent = monitor.end_request() if monitor else None
            answer = result.answer if args.image else result
            if len(timer.tokens) != len(answer.token_ids):
                raise RuntimeError('Token timestamp count differs from generated ID count')
            rows.append(dict(run=number, pipeline_latency_ms=(end-start)/1e6, **timer.metrics(),
                             finish_reason=answer.finish_reason, answer=answer.text,
                             token_ids=json.dumps(answer.token_ids), backend='TT-XLA', dtype='bfloat16', experiment=mode,
                             telemetry_enabled=bool(telemetry_enabled), process_cpu_percent_one_core=cpu_percent,
                             telemetry_file=str(telemetry_path) if monitor else None,
                             interval_ms=args.interval_ms, device_interval_ms=args.device_interval_ms,
                             tt_smi=args.tt_smi, metrics_url=args.metrics_url,
                             versions=json.dumps(versions), model=args.model, question=args.question,
                             image=args.image, warmup=args.warmup, max_new_tokens=args.max_new_tokens,
                             alignment=args.alignment, adapters=args.adapters, rag_query_mode=args.rag_query_mode,
                             retrieved_entry_id=result.retrieved_entry_id if args.image else None))
    finally:
        if monitor:
            monitor.finish(telemetry_path)
            print(f"Saved telemetry to {telemetry_path}")
    if monitor:
        for row in rows:
            row['telemetry_summary'] = json.dumps(monitor.summary(row['run']))
        errors = sum(sample['status'] != 'ok' for sample in monitor.rows)
        print(f"Telemetry status/error records {errors}. Inspect the telemetry CSV status column.")
    with args.output.open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} requests to {args.output}")
    print(f"Median pipeline latency {median(row['pipeline_latency_ms'] for row in rows):.2f} ms")


if __name__ == '__main__':
    main()
