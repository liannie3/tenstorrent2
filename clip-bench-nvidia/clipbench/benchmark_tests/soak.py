# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            soak.py
# CREATED:         11/10/2026
# REVISION:        1.0
# ======================================= #

import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .. import data
from .. import model as model_lib

## 1000 ms is the tegrastats default, which gives ~8 samples for a views=1
## sub-test. 100 ms is enough to see the clock ramp.
INTERVAL_MS = 100


## tegrastats behaves like a singleton -- a stale instance makes a new one print
## nothing. Stopping first is cheap and fixes that.
def _stop():
    try:
        subprocess.run(['tegrastats', '--stop'],
                       capture_output=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


## One sub-test: every frame, at a single view count.
## timings is unused -- the samples are the data, and they are in the log.
## Kept so every test has the same signature.
def run(config, encoder, processor, projector, views_paths, views, timings):

    ## Loads parameters which are initialized in the jetson.yaml file
    device = config['run']['device']                               # config['run']['device'] returns cuda
    dtype = model_lib.resolve_dtype(config['run']['precision'])    # Goes to model.py and makes sure config dtype is allowed
    max_frames = config['run'].get('max_frames')                   # Caps the iterations
    images_seen = 0                                                # Image counter

    num_frames = data.num_frames(views_paths)                      # Iterations available, set by the shortest view
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    ## One log per view count, beside the other tests' output
    out_dir = Path(config['output']['results_dir']) / 'hardware_perf'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'tegrastats_v{views}.log'
    interval = config['run'].get('tegrastats_interval_ms', INTERVAL_MS)

    _stop()
    try:
        ## --logfile rather than a pipe, so nothing here has to drain stdout
        sampler = subprocess.Popen(['tegrastats', '--interval', str(interval),
                                    '--logfile', str(log_path)])
    except FileNotFoundError:
        warnings.warn('tegrastats not on PATH, so no hardware samples - run this on '
                      'the Orin', RuntimeWarning, stacklevel=2)
        sampler = None
    else:
        print(f'  tegrastats  : every {interval} ms -> {log_path}')

    ## Everything the hand-written stages need, pulled out once before the loop
    vision = encoder.vision_model                       # CLIPVisionTransformer inside CLIPVisionModel
    emb = vision.embeddings
    cfg = encoder.config
    hidden = cfg.hidden_size                            # 1024 for ViT-L
    num_heads = cfg.num_attention_heads                 # 16
    head_dim = hidden // num_heads                      # 64
    eps = cfg.layer_norm_eps
    if cfg.hidden_act != 'quick_gelu':
        raise ValueError(f'MLP below is quick_gelu, checkpoint uses {cfg.hidden_act!r}')

    short_edge = processor.size['shortest_edge']        # 336
    crop_h = processor.crop_size['height']              # 336
    crop_w = processor.crop_size['width']               # 336
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)

    layer_index = config['model']['layer_index']
    keep_cls = config['model']['keep_cls']

    try:
        with torch.inference_mode():
            for i, frame_paths in enumerate(data.iter_frames(views_paths)):
                if i >= num_frames:
                    break

                ## One sub-iteration per view, no synchronize between them
                for path in frame_paths:

                    ## Untimed on purpose -- tegrastats is the only instrument here

                    # ---------- Decode ----------
                    ## JPEG bytes -> RGB, so CLIP always sees 3 channels
                    image = Image.open(path).convert('RGB')

                    # ---------- Preprocess ----------
                    ## Resize shortest edge to 336, keeping aspect (1600x900 -> 597x336)
                    w, h = image.size
                    if w <= h:
                        new_w, new_h = short_edge, int(short_edge * h / w)
                    else:
                        new_w, new_h = int(short_edge * w / h), short_edge
                    image = image.resize((new_w, new_h), resample=processor.resample)

                    ## Center crop to 336x336
                    left = (new_w - crop_w) // 2
                    top = (new_h - crop_h) // 2
                    image = image.crop((left, top, left + crop_w, top + crop_h))

                    ## uint8 HWC -> float CHW in [0, 1], then per-channel normalize
                    pixels = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float()
                    pixels = pixels * processor.rescale_factor                    # 1/255
                    pixels = (pixels - mean) / std
                    pixel_values = pixels.unsqueeze(0).to(device=device, dtype=dtype)  # (1, 3, 336, 336)

                    # ---------- Encoder: embeddings ----------
                    ## 14x14 conv with stride 14 cuts the image into a 24x24 grid of patches
                    x = F.conv2d(pixel_values, emb.patch_embedding.weight,
                                 emb.patch_embedding.bias, stride=emb.patch_size)  # (1, 1024, 24, 24)
                    x = x.flatten(2).transpose(1, 2)                                # (1, 576, 1024)

                    ## Prepend the learned CLS token, add learned position embeddings
                    cls = emb.class_embedding.expand(x.shape[0], 1, -1)
                    x = torch.cat([cls, x], dim=1)                                  # (1, 577, 1024)
                    x = x + emb.position_embedding.weight                           # positions 0..576 in order

                    x = F.layer_norm(x, (hidden,), vision.pre_layrnorm.weight,
                                     vision.pre_layrnorm.bias, eps)
                    hidden_states = [x]                                             # same layout as output_hidden_states

                    # ---------- Encoder: 24 transformer layers ----------
                    for layer in vision.encoder.layers:
                        attn = layer.self_attn
                        B, T, _ = x.shape

                        ## Self-attention, pre-norm
                        h = F.layer_norm(x, (hidden,), layer.layer_norm1.weight, layer.layer_norm1.bias, eps)
                        q = F.linear(h, attn.q_proj.weight, attn.q_proj.bias)
                        k = F.linear(h, attn.k_proj.weight, attn.k_proj.bias)
                        v = F.linear(h, attn.v_proj.weight, attn.v_proj.bias)

                        ## (B, T, 1024) -> (B, 16 heads, T, 64)
                        q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
                        k = k.view(B, T, num_heads, head_dim).transpose(1, 2)
                        v = v.view(B, T, num_heads, head_dim).transpose(1, 2)

                        ## softmax(q k^T / sqrt(64)) v -- no mask, every patch sees every patch
                        h = F.scaled_dot_product_attention(q, k, v)
                        h = h.transpose(1, 2).reshape(B, T, hidden)
                        h = F.linear(h, attn.out_proj.weight, attn.out_proj.bias)
                        x = x + h

                        ## MLP, pre-norm: 1024 -> 4096 -> quick_gelu -> 1024
                        h = F.layer_norm(x, (hidden,), layer.layer_norm2.weight, layer.layer_norm2.bias, eps)
                        h = F.linear(h, layer.mlp.fc1.weight, layer.mlp.fc1.bias)
                        h = h * torch.sigmoid(1.702 * h)
                        h = F.linear(h, layer.mlp.fc2.weight, layer.mlp.fc2.bias)
                        x = x + h

                        hidden_states.append(x)

                    ## Pooled CLS output -- unused, but encoder() does it, so the load matches
                    pooled = F.layer_norm(x[:, 0, :], (hidden,), vision.post_layernorm.weight,
                                          vision.post_layernorm.bias, eps)

                    # ---------- Feature select ----------
                    ## layer_index -2 = second-to-last layer (LLaVA-1.5 style)
                    features = hidden_states[layer_index]
                    if not keep_cls:
                        features = features[:, 1:, :]

                    # ---------- Projector ----------
                    ## 1024 -> 2048 -> GELU -> 2048
                    h = F.linear(features, projector.fc1.weight, projector.fc1.bias)
                    h = F.gelu(h)
                    embeddings = F.linear(h, projector.fc2.weight, projector.fc2.bias)

                    images_seen += 1

        ## Drains the queue so the last kernels land inside the sampled window
        torch.cuda.synchronize()
    finally:
        ## in a finally so a SystemExit mid-sweep does not leave it running
        if sampler is not None:
            _stop()
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()

    ## Shapes are the same every iteration, so the last batch describes them all
    if images_seen:
        print('  pixel_values:', tuple(pixel_values.shape))
        print('  features    :', tuple(features.shape))
        print('  projected   :', tuple(embeddings.shape))

    return images_seen
