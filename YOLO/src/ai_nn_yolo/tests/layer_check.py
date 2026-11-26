"""On-card debug: PCC of every backbone module (layers 0-9) against PyTorch, so a failing layer can be named.

  python tests/layer_check.py                 on the p150a (uses config.py's WEIGHTS_DTYPE / MATH_FIDELITY)
  python tests/layer_check.py --fake          on the CPU with tests/fake_ttnn.py (must print 1.00000 everywhere)

For each layer it prints two PCCs:
  isolated  the module run on the exact PyTorch input of that layer (uploaded to the card): that layer's own error
  chained   the module run on the card's own previous output, like backbone() does: error built up so far
For C2f layers it also checks the pieces inside the block (cv1_a, cv1_b, each bottleneck, the concat, cv2) on the exact input.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser()
parser.add_argument("--fake", action="store_true", help="run on the CPU with tests/fake_ttnn.py")
args = parser.parse_args()
if args.fake:
    import fake_ttnn
    sys.modules["ttnn"] = fake_ttnn.as_module()

import ttnn  # noqa: E402

import config  # noqa: E402
from main import load_image_path, load_model  # noqa: E402
from run import load_images  # noqa: E402
from ttnn_yolov8n import TtDetectionModel  # noqa: E402
from ttnn_yolov8n_utils import custom_preprocessor  # noqa: E402
from verify import pcc, reference_forward, reference_input  # noqa: E402


def upload(t):
    """NCHW torch -> flattened NHWC [1, 1, N*H*W, C] bf16 TILE tensor in L1 (what a module receives between layers)."""
    n, c, h, w = t.shape
    x = t.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c).to(torch.bfloat16)
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


def back(t, n, h, w, c):
    """Card tensor [1, 1, N*H*W, >=C] -> NCHW float32 torch."""
    x = ttnn.to_torch(t).float().reshape(n, h, w, -1)[..., :c]
    return x.permute(0, 3, 1, 2).contiguous()


def show(name, value):
    flag = "" if value >= 0.99 else ("   <-- WARN" if value >= 0.97 else "   <-- FAIL")
    print(f"  {name:<34} {value:9.5f}{flag}")


def c2f_parts(tt_c2f, torch_c2f, x_ref, prefix):
    """Run TtC2f.__call__ step by step on the exact input and compare each piece with PyTorch's C2f.forward."""
    n, _, h, w = x_ref.shape
    with torch.inference_mode():
        y_ref = list(torch_c2f.cv1(x_ref).chunk(2, 1))
        for m in torch_c2f.m:
            y_ref.append(m(y_ref[-1]))
        cat_ref = torch.cat(y_ref, 1)
        out_ref = torch_c2f.cv2(cat_ref)

    x = upload(x_ref)
    cv1_a, oh, ow = tt_c2f.cv1_a(x, h, w)
    cv1_b, oh, ow = tt_c2f.cv1_b(x, h, w)
    cv1_a = ttnn.to_memory_config(cv1_a, ttnn.L1_MEMORY_CONFIG)
    cv1_b = ttnn.to_memory_config(cv1_b, ttnn.L1_MEMORY_CONFIG)
    half = y_ref[0].shape[1]
    show(f"{prefix} cv1_a", pcc(back(cv1_a, n, oh, ow, half), y_ref[0]))
    show(f"{prefix} cv1_b", pcc(back(cv1_b, n, oh, ow, half), y_ref[1]))
    y = [cv1_a, cv1_b]
    for i, block in enumerate(tt_c2f.bottleneck_modules):
        z = block(y[-1], oh, ow)
        y.append(z)
        show(f"{prefix} bottleneck m.{i}", pcc(back(z, n, oh, ow, half), y_ref[2 + i]))
    if not tt_c2f.shortcut:
        for i in range(2, len(y)):
            y[i] = ttnn.sharded_to_interleaved(y[i], ttnn.L1_MEMORY_CONFIG)
    cat = ttnn.concat(y, 3)
    show(f"{prefix} concat", pcc(back(cat, n, oh, ow, cat_ref.shape[1]), cat_ref))
    out, oh, ow = tt_c2f.cv2(cat, oh, ow)
    show(f"{prefix} cv2 (block output)", pcc(back(out, n, oh, ow, out_ref.shape[1]), out_ref))


if __name__ == "__main__":
    input_hw = tuple(config.INPUT_HW)
    if args.fake:
        from ultralytics.utils import ASSETS
        import cv2
        images = [cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), (1600, 900))]
    else:
        images = load_images(load_image_path(config.IMG_DIR)[:1])
    torch_model = load_model()
    x_ref = reference_input(images, input_hw)
    _, ref = reference_forward(torch_model, x_ref)

    device = ttnn.open_device(device_id=config.DEVICE_ID, l1_small_size=config.L1_SMALL_SIZE)
    try:
        params = custom_preprocessor(device, torch_model.state_dict(), input_hw[0], input_hw[1],
                                     bfloat8=config.WEIGHTS_DTYPE == "bfp8")
        #two separate model objects: conv2d prepares each conv's weights for the input layout of its first call, so the
        #isolated (L1 interleaved input) and chained (sharded input) runs must not share prepared weights
        build = lambda: TtDetectionModel(device, params, res=input_hw, batch_size=1,  # noqa: E731
                                         weights_dtype=config.WEIGHTS_DTYPE, math_fidelity=config.MATH_FIDELITY)
        iso, chain = build(), build()
        names = ["conv_0", "conv_1", "c2f_2", "conv_3", "c2f_4", "conv_5", "c2f_6", "conv_7", "c2f_8", "sppf_9"]
        print(f"layer check: {config.WEIGHTS_DTYPE} + {config.MATH_FIDELITY}, input {input_hw[0]}x{input_hw[1]}, batch 1")

        # ---------------- chained (what backbone() does)
        print("\nCHAINED (each module gets the card's previous output)")
        raw = torch.from_numpy(np.ascontiguousarray(np.stack(
            [__import__("ultralytics").data.augment.LetterBox(new_shape=input_hw, auto=False, stride=32)(image=im) for im in images]
        )[..., ::-1].transpose(0, 3, 1, 2))).to(torch.bfloat16)
        x = chain.prepare_input(ttnn.from_torch(raw, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                                memory_config=ttnn.DRAM_MEMORY_CONFIG))
        h, w = input_hw
        for i, name in enumerate(names):
            x, h, w = getattr(chain, name)(x, h, w)
            c = ref[i].shape[1]
            show(f"layer {i} {name}", pcc(back(x, 1, h, w, c), ref[i]))

        # ---------------- isolated (exact PyTorch input for every layer)
        print("\nISOLATED (each module gets the exact PyTorch input of that layer)")
        for i, name in enumerate(names):
            if i == 0:
                continue  #layer 0's input is the image itself, already covered by the chained row
            inp = ref[i - 1]
            out, h, w = getattr(iso, name)(upload(inp), inp.shape[2], inp.shape[3])
            show(f"layer {i} {name}", pcc(back(out, 1, h, w, ref[i].shape[1]), ref[i]))

        # ---------------- inside every C2f, exact input
        print("\nINSIDE EACH C2f (exact PyTorch input)")
        parts = build()
        for i in (2, 4, 6, 8):
            c2f_parts(getattr(parts, f"c2f_{i}"), torch_model.model[i], ref[i - 1], f"layer {i}")
    finally:
        ttnn.close_device(device)
