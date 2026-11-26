"""Run the whole harness on the CPU with tests/fake_ttnn.py standing in for ttnn (no card needed).

  python tests/run_cpu_check.py                         verify + latency at batch 1 and 2, exact fp32 math (PCC must be ~1.0)
  FAKE_TTNN_DTYPE=bfp8 python tests/run_cpu_check.py    same, with bfloat8_b rounding to estimate the accuracy cost

Images: ultralytics' sample images resized to 1600x900 (the CAM_FRONT size), or --img-dir. Results go to a temp folder,
not benchmarking/.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)) #the harness folder
sys.path.insert(0, str(HERE))

import fake_ttnn  # noqa: E402

sys.modules["ttnn"] = fake_ttnn.as_module()

import config  # noqa: E402
import main as harness  # noqa: E402


def sample_dir(tmp):
    from ultralytics.utils import ASSETS
    out = Path(tmp) / "imgs"
    out.mkdir()
    for p in sorted(ASSETS.glob("*.jpg")):
        cv2.imwrite(str(out / p.name), cv2.resize(cv2.imread(str(p)), (1600, 900)))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        config.IMG_DIR = Path(args.img_dir) if args.img_dir else sample_dir(tmp)
        config.CSV_PATH = Path(tmp) / "stage_timings_tt.csv"
        config.VERIFY_CSV_PATH = Path(tmp) / "verify_tt.csv"
        config.WARMUP_ITERS = 1
        print(f"fake ttnn mode: {fake_ttnn.MODE}, images from {config.IMG_DIR}")
        for bs in args.batch_sizes:
            print(f"\n================ batch {bs} ================")
            harness.main(argparse.Namespace(setting="holistic", batch_size=bs, step=1))
