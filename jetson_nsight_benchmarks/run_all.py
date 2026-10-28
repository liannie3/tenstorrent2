"""Run the same workload sequentially through all three Nsight tests."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='models/tinyllama')
    p.add_argument('--question',default='Explain the difference between a car and a pedestrian.')
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--no-cpu-sampling',action='store_true')
    p.add_argument('--output-dir',default='results/'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    a=p.parse_args()
    for mode in ('latency','cpu','gpu'):
        cmd=[sys.executable,'-B',str(Path(__file__).with_name(f'benchmark_{mode}.py')),
             '--text-only','--model',a.model,'--question',a.question,
             '--runs',str(a.runs),'--warmup',str(a.warmup),'--max-new-tokens',str(a.max_new_tokens),
             '--output',str(Path(a.output_dir)/(mode+'.nsys-rep'))]
        if mode=='cpu' and a.no_cpu_sampling: cmd+=['--no-cpu-sampling']
        result=subprocess.run(cmd,check=False)
        if result.returncode: return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
