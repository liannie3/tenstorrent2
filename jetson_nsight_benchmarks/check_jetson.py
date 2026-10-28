"""Check this Python environment before downloading or benchmarking."""
import importlib
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    failed = False
    print('Python:',sys.executable,sys.version)
    print('Machine:',platform.machine())
    release = Path('/etc/nv_tegra_release')
    if release.exists(): print('Jetson release:',release.read_text().strip())
    for name in ['torch','transformers','huggingface_hub','tokenizers','safetensors','sentencepiece','numpy']:
        try:
            module = importlib.import_module(name)
            print(name,getattr(module,'__version__','installed'))
        except Exception as exc:
            print('FAIL',name,str(exc)); failed=True
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable in this Python environment')
        print('CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0))
        with torch.cuda.nvtx.range('setup_check'):
            x=torch.ones(4,device='cuda'); y=x+x
            torch.cuda.synchronize()
            assert y.sum().item()==8
        print('CUDA operation and NVTX: PASS')
    except Exception as exc:
        print('FAIL CUDA/NVTX:',exc); failed=True
    nsys=shutil.which('nsys')
    if nsys:
        result=subprocess.run([nsys,'--version'],check=False)
        failed |= result.returncode != 0
    else:
        print('FAIL: nsys is not on PATH'); failed=True
    print('Ready for a capture smoke test.' if not failed else 'Resolve the failures above before benchmarking.')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
