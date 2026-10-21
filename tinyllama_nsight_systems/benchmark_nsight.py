"""Optional Nsight launcher. Retains only the final .nsys-rep report."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def launch(args):
    target = Path(args.benchmark).resolve()
    if target.suffix != '.nsys-rep':
        raise ValueError('--benchmark must name a .nsys-rep file.')
    if target.exists():
        raise FileExistsError('Choose a new report name; existing captures are preserved.')
    nsys = shutil.which('nsys')
    if not nsys:
        raise RuntimeError('Install NVIDIA Nsight Systems and add nsys to PATH before benchmarking.')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Preserve identical inputs and adapter selection in the profiled child.
    child = [sys.executable, 
             '-B', str(Path(__file__).with_name('run.py').resolve()),
             '--profile-worker', 
             '--model', args.model, 
             '--question', args.question,
             '--max-new-tokens', 
             str(args.max_new_tokens), 
             '--runs', str(args.runs),
             '--warmup', str(args.warmup), 
             '--interval-ms', str(args.interval_ms),
             '--rag-query-mode', args.rag_query_mode]
    if args.text_only:
        child.append('--text-only')
    for flag, value in [('--image', args.image), ('--alignment', args.alignment),
                        ('--adapters', args.adapters), ('--clip-embedding-id', args.clip_embedding_id)]:
        if value is not None:
            child.extend([flag, value])
    # Keep tool intermediates temporary. No JSON, CSV or text-log exports.
    with tempfile.TemporaryDirectory(prefix='tinyllama-nsys-') as temporary:
        prefix = Path(temporary) / 'capture'
        command = [nsys, 'profile', 
                   '--trace=cuda,nvtx,osrt', 
                   '--stats=false',
                   '--cuda-memory-usage=true', 
                   '--capture-range=cudaProfilerApi',
                   '--capture-range-end=stop', 
                   '--output=' + str(prefix),
                   '--sample=process-tree' if args.cpu_sampling else '--sample=none',
                   '--cpuctxsw=process-tree' if args.cpu_sampling else '--cpuctxsw=none']
        if args.gpu_metrics_device:
            command.append('--gpu-metrics-device=' + args.gpu_metrics_device)
        result = subprocess.run(command + child, check=False)
        native = prefix.with_suffix('.nsys-rep')
        if native.is_file() and native.stat().st_size:
            with native.open('rb') as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            print(f'Saved: {target}', file=sys.stderr)
            if result.returncode:
                print('Capture is incomplete; inspect the terminal error.', file=sys.stderr)
        else:
            print('No native report was produced; inspect the Nsight error.', file=sys.stderr)
            return result.returncode or 1
        return result.returncode
