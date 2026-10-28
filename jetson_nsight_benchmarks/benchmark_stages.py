"""Launch separate latency/CPU/GPU experiments, all through Nsight Systems."""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def parser_for(mode=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test', choices=['latency','cpu','gpu'], default=mode, required=mode is None)
    p.add_argument('--model', required=True)
    p.add_argument('--question', default='Explain the difference between a car and a pedestrian.')
    p.add_argument('--output', required=True, help='New .nsys-rep filename')
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--text-only', action='store_true')
    p.add_argument('--image')
    p.add_argument('--alignment')
    p.add_argument('--adapters')
    p.add_argument('--rag-query-mode', choices=['tinyllama_text','clip'], default='tinyllama_text')
    p.add_argument('--clip-embedding-id')
    p.add_argument('--no-cpu-sampling', action='store_true', help='CPU fallback when native sampling/scheduling is unavailable')
    p.add_argument('--gpu-metrics-device', help='Optional GPU test hardware metrics; platform dependent')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return p


def profile_options(args):
    sample = args.test == 'cpu' and not args.no_cpu_sampling
    trace = 'cuda,nvtx,osrt' if args.test == 'cpu' else 'cuda,nvtx'
    opts = ['--trace='+trace, '--sample='+('process-tree' if sample else 'none'),
            '--cpuctxsw='+('process-tree' if sample else 'none'), '--stats=false',
            '--capture-range=cudaProfilerApi', '--capture-range-end=stop']
    if args.gpu_metrics_device:
        opts += ['--gpu-metrics-device='+args.gpu_metrics_device]
    return opts


def launch(args):
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(f'Choose a new report filename: {target}')
    nsys = shutil.which('nsys')
    if not nsys:
        raise RuntimeError('nsys is not on PATH. Run python check_jetson.py.')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Rebuild arguments explicitly so wrapper scripts and paths with spaces work.
    child = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker']
    for name in ('test','model','question','output','max_new_tokens','warmup','runs',
                 'image','alignment','adapters','rag_query_mode','clip_embedding_id'):
        value = getattr(args,name)
        if value is not None:
            child += ['--'+name.replace('_','-'),str(value)]
    if args.text_only:
        child += ['--text-only']
    # Keep only the native report. Nsight intermediary files stay temporary.
    with tempfile.TemporaryDirectory(prefix='tinyllama-nsys-') as temp:
        prefix = Path(temp)/'capture'
        command = [nsys,'profile',*profile_options(args),'--output='+str(prefix),*child]
        result = subprocess.run(command,check=False)
        native = prefix.with_suffix('.nsys-rep')
        if result.returncode:
            print('Capture failed. Inspect Nsight output. CPU sampling may need system permissions; '
                  'retry explicitly with --no-cpu-sampling if appropriate.',file=sys.stderr)
            return result.returncode
        if not native.is_file() or not native.stat().st_size:
            raise RuntimeError('Nsight produced no report.')
        with native.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src,dst)
    print(f'Saved {target}')
    return 0


def worker(args):
    import torch
    from tinyllama_stage import TinyLlamaStage
    from stage_observer import StageObserver
    llm = TinyLlamaStage(args.model,alignment_manifest=args.alignment)
    pipeline = None
    if not args.text_only:
        from pipeline import AVPipeline
        module,factory = args.adapters.split(':',1)
        components = getattr(importlib.import_module(module),factory)(llm)
        pipeline = AVPipeline(llm,**components,rag_query_mode=args.rag_query_mode,
                              embedding_id=args.clip_embedding_id)
    def call(observer=None):
        if pipeline is None:
            return llm.answer(args.question,max_new_tokens=args.max_new_tokens,benchmark=observer)
        return pipeline.run(args.image,args.question,max_new_tokens=args.max_new_tokens,benchmark=observer)
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize(llm.device)
    answers = []
    torch.cuda.profiler.start()
    try:
        torch.cuda.nvtx.mark('experiment '+json.dumps({
            'test':args.test,'model':args.model,'question':args.question,
            'runs':args.runs,'warmup':args.warmup,'max_new_tokens':args.max_new_tokens,
            'torch':str(torch.__version__),'cuda':torch.version.cuda,
            'device':torch.cuda.get_device_name(llm.device)}))
        for i in range(args.runs):
            observer = StageObserver(args.test,llm.device,torch)
            torch.cuda.synchronize(llm.device)
            with observer.span(f'request_{i+1}'):
                answers.append(call(observer))
                torch.cuda.synchronize(llm.device)
    finally:
        torch.cuda.synchronize(llm.device)
        torch.cuda.profiler.stop()
    # Avoid terminal output inside capture.
    for answer in answers:
        print(json.dumps(asdict(answer),ensure_ascii=False))
    return 0


def main(mode=None):
    p = parser_for(mode)
    args = p.parse_args()
    if mode and args.test != mode:
        p.error('Use the matching benchmark script.')
    if Path(args.output).suffix != '.nsys-rep':
        p.error('--output must end with .nsys-rep')
    if args.runs<1 or args.warmup<0 or args.max_new_tokens<1 or not args.question.strip():
        p.error('Use runs/tokens >=1, warmup >=0 and a nonempty question.')
    if args.text_only and any((args.image,args.alignment,args.adapters)):
        p.error('--text-only cannot include image, alignment or adapters.')
    if not args.text_only and not all((args.image,args.alignment,args.adapters)):
        p.error('Use --text-only or supply --image, --alignment and --adapters.')
    if args.adapters and (':' not in args.adapters or not all(args.adapters.split(':',1))):
        p.error('--adapters must be module:factory.')
    if args.gpu_metrics_device and args.test != 'gpu':
        p.error('--gpu-metrics-device is for the GPU test.')
    if args.no_cpu_sampling and args.test != 'cpu':
        p.error('--no-cpu-sampling is for the CPU test.')
    return worker(args) if args.worker else launch(args)


if __name__ == '__main__':
    raise SystemExit(main())
