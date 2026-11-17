# tt_clip/xla

TT-XLA version of the CLIP ViT-L/14-336 vision encoder: the stock HuggingFace PyTorch
model compiled for the Tenstorrent card through TT-XLA, instead of the hand-ported
TT-NN model in the parent `tt_clip/` folder.

Used as the "out-of-the-box compiler" comparison against the TT-NN port.

- `clip_xla.py` - builds the model: HF CLIPVisionModel cut to 23 layers, CLS dropped,
  cast to bf16, `model.compile(backend="tt")`, moved to the XLA device.
- `test.py` - same structure as `../test.py`: PCC vs HF fp32 on CPU for cat/dog, then the
  frames x views timing loop. `--opt-level {0,1,2}` and `--trace` set the TT-XLA compile options.

Install (separate venv from TT-NN is safest):

    pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
    tt-forge-install
    pip install transformers torchvision pillow
