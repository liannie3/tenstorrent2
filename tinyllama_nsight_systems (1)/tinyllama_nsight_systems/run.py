"""Run normally; add --benchmark FILE.nsys-rep only when profiling is wanted.

Normal text example:
  python -B run.py --model models/tinyllama --text-only --question "What is a car?"
Pipeline example (requires the team's real adapters and trained alignment):
  python -B run.py --model models/tinyllama --image scene.jpg \
    --alignment alignment.json --adapters team_adapters:build \
    --question "Describe this scene."
Append --benchmark results/test_01.nsys-rep to either command for Nsight.
Only the native report is saved; answers and tool messages go to the terminal.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from dataclasses import asdict
import importlib
import json
import logging


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='Existing local TinyLlama checkpoint')
    parser.add_argument('--question', '--prompt', dest='question', required=True)
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--image')
    parser.add_argument('--alignment', help='Actual trained visual-alignment manifest')
    parser.add_argument('--adapters', help='Team module:factory returning yolo, vision and optional rag objects')
    parser.add_argument('--rag-query-mode', choices=['tinyllama_text', 'clip'], default='tinyllama_text')
    parser.add_argument('--clip-embedding-id')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--benchmark', metavar='FILE.nsys-rep', help='Optional Nsight capture; absent means normal inference')
    parser.add_argument('--runs', type=int, default=1, help='Measured repetitions in benchmark mode only')
    parser.add_argument('--warmup', type=int, default=1, help='Uncaptured warm-up requests in benchmark mode only')
    parser.add_argument('--interval-ms', type=int, default=100, help='CPU/RAM NVTX sampling interval in benchmark mode')
    parser.add_argument('--gpu-metrics-device', help='Optional native Nsight GPU metrics, e.g. all or 0')
    parser.add_argument('--cpu-sampling', action='store_true', help='Optional native CPU backtraces; needs platform support')
    parser.add_argument('--profile-worker', action='store_true', help=argparse.SUPPRESS)
    return parser


def execute(args):
    # Load models once and reuse the same call path in normal and profiled runs.
    from tinyllama_stage import TinyLlamaStage
    llm = TinyLlamaStage(args.model, alignment_manifest=args.alignment)
    pipeline = None
    if not args.text_only:
        from pipeline import AVPipeline
        module, factory = args.adapters.split(':', 1)
        components = getattr(importlib.import_module(module), factory)(llm)
        pipeline = AVPipeline(llm, **components, rag_query_mode=args.rag_query_mode,
                              embedding_id=args.clip_embedding_id)

    def call_once(observer=None):
        # Omitting the observer avoids hooks, timers and monitoring in normal mode.
        optional = {'benchmark': observer} if observer is not None else {}
        if pipeline is None:
            return llm.answer(args.question, max_new_tokens=args.max_new_tokens, **optional)
        return pipeline.run(args.image, args.question, max_new_tokens=args.max_new_tokens, **optional)

    if not args.profile_worker:
        result = call_once()
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0

    import torch
    from nsight_buffer import NsightBuffer
    # Loading/warm-up are excluded. Warm-up repeats the real pipeline and adapters.
    for _ in range(args.warmup):
        call_once()
    torch.cuda.synchronize(llm.device)
    torch.cuda.reset_peak_memory_stats(llm.device)
    answers = []
    torch.cuda.profiler.start()#############################
    try:
        for index in range(args.runs):
            observer = NsightBuffer(args.interval_ms)
            with observer.pipeline_scope(llm, f'request_{index + 1}'):
                answers.append(call_once(observer))
        torch.cuda.synchronize(llm.device)
        torch.cuda.nvtx.mark(
            f'pytorch_gpu_memory peak_allocated_bytes={torch.cuda.max_memory_allocated(llm.device)} '
            f'peak_reserved_bytes={torch.cuda.max_memory_reserved(llm.device)} '
            f'allocated_bytes={torch.cuda.memory_allocated(llm.device)} '
            f'reserved_bytes={torch.cuda.memory_reserved(llm.device)}')
    finally:
        torch.cuda.synchronize(llm.device)
        torch.cuda.profiler.stop()###################################
    # Printing happens after capture, never inside the measured pipeline.
    for answer in answers:
        print(json.dumps(asdict(answer), ensure_ascii=False))
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.text_only and any((args.image, args.alignment, args.adapters)):
        parser.error('--text-only cannot be combined with image/alignment/adapters.')
    if not args.text_only and not all((args.image, args.alignment, args.adapters)):
        parser.error('Pipeline mode requires --image, --alignment and --adapters; or use --text-only.')
    if args.adapters and (':' not in args.adapters or not all(args.adapters.split(':', 1))):
        parser.error('--adapters must be module:factory')
    if min(args.runs, args.max_new_tokens) < 1 or args.warmup < 0 or args.interval_ms < 20 or not args.question.strip():
        parser.error('Use positive counts, warmup >=0, interval >=20 ms and a nonempty question.')
    if not args.benchmark and not args.profile_worker and (args.runs != 1 or args.gpu_metrics_device or args.cpu_sampling):
        parser.error('--runs, --gpu-metrics-device and --cpu-sampling require --benchmark.')
    try:
        if args.benchmark and not args.profile_worker:
            # The parent launches nsys before importing/loading any ML models.
            from benchmark_nsight import launch
            return launch(args)
        return execute(args)
    except Exception:
        logging.exception('Run failed; see the error above.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
