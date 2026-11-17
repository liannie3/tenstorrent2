"""Read-only environment check; no model, timers or benchmarks."""
import platform
import subprocess
import sys
from pathlib import Path

print('Python:', sys.version)
print('Architecture:', platform.machine())
for name in ('/etc/nv_tegra_release', '/etc/os-release'):
    if Path(name).exists():
        print(name + ':\n' + Path(name).read_text())
try:
    result = subprocess.run(['dpkg-query', '-W', 'nvidia-jetpack', 'nvidia-l4t-core'],
                            text=True, capture_output=True, check=False)
    print(result.stdout, result.stderr)
except FileNotFoundError:
    print('dpkg-query unavailable.')
if sys.version_info[:2] != (3, 10):
    print('Select a compatible Python 3.10 environment for this package.')
try:
    import torch
    print('PyTorch:', torch.__version__, 'built CUDA:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    print('GPU:', torch.cuda.get_device_name(0))
    x = torch.ones((2, 2), device='cuda', dtype=torch.float16)
    assert (x @ x).cpu().tolist() == [[2., 2.], [2., 2.]]
    print('CUDA FP16 operation passed.')
except Exception as exc:
    print('PyTorch/CUDA not ready:', exc)
    raise SystemExit(1)
