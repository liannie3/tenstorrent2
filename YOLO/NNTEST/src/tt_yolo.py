"""TT-NN YOLOv8, built module-by-module from the fused ultralytics DetectionModel.

Every ultralytics block in `model.model` (Conv, C2f, SPPF, Upsample, Concat, Detect) gets a TT counterpart that keeps the
block's `i`, `f` and `type` attributes, so `stages.run_layers` walks the TT model exactly like the CUDA version walks
`model.model`, and the backbone / neck / head split (layers 0-9 / 10-21 / 22) is unchanged.

Tensors between layers are `Feat` objects: a TT-NN tensor in the flattened NHWC form [1, 1, N*H*W, C] that ttnn.conv2d
consumes and produces, plus the (n, h, w, c) it represents. TT convs are channels-last, so there is no NCHW permute anywhere
inside the network.

Weights come straight from the fused torch model (BatchNorm already folded into each conv's weight and bias). They are
handed to ttnn.conv2d as host tensors on the first call; conv2d lays them out for the card and returns the device copies,
which are cached per input shape and reused on every later call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import ttnn

IMG_CHANNELS_PADDED = 16  #the 3-channel image is zero-padded to 16 channels before the first conv (same as tt-metal's YOLO demos)
BOX_COLS = 32             #decoded xywh are produced in a 32-wide (one tile) block; only the first 4 columns are meaningful
TILE = 32


@dataclass
class Feat:
    """A feature map on the card: `t` is the flattened NHWC ttnn tensor [1, 1, n*h*w, c]."""

    t: object
    n: int
    h: int
    w: int
    c: int

    def to_torch_nchw(self):
        """Copy to host as a float32 NCHW torch tensor (used by verification)."""
        x = ttnn.to_torch(self.t).float()
        x = x.reshape(self.n, self.h, self.w, -1)[..., : self.c]
        return x.permute(0, 3, 1, 2).contiguous()


########################################################################################################################################################################
#   helpers                                                                                                                                                            #
########################################################################################################################################################################


def _activation(module):
    """Map an ultralytics Conv's activation to the activation fused into ttnn.conv2d (None = identity)."""
    act = getattr(module, "act", None)
    if act is None or isinstance(act, nn.Identity):
        return None
    if isinstance(act, nn.SiLU):
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    if isinstance(act, nn.ReLU):
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    raise NotImplementedError(f"activation {type(act).__name__} has no TT-NN mapping yet")


def _conv_weights(module):
    """(nn.Conv2d, weight, bias) of an ultralytics Conv or a plain nn.Conv2d, with BatchNorm folded in if the model is unfused."""
    conv = module if isinstance(module, nn.Conv2d) else module.conv
    w = conv.weight.detach().float().clone()
    b = conv.bias.detach().float().clone() if conv.bias is not None else torch.zeros(w.shape[0])
    bn = getattr(module, "bn", None)
    if isinstance(bn, nn.BatchNorm2d):  #unfused model: fold BN exactly like DetectionModel.fuse() does
        scale = bn.weight.detach().float() / torch.sqrt(bn.running_var.float() + bn.eps)
        w = w * scale[:, None, None, None]
        b = (b - bn.running_mean.float()) * scale + bn.bias.detach().float()
    return conv, w, b


def concat_channels(ctx, feats):
    """Channel concat (torch.cat(dim=1) in NCHW == last dim here). TILE concat needs 32-aligned widths, otherwise row-major."""
    n, h, w = feats[0].n, feats[0].h, feats[0].w
    for g in feats:
        if (g.n, g.h, g.w) != (n, h, w):
            raise ValueError(f"concat of mismatched feature maps {(g.n, g.h, g.w)} vs {(n, h, w)}")
    tile_ok = all(g.t.layout == ttnn.TILE_LAYOUT and g.c % TILE == 0 for g in feats)
    if tile_ok:
        parts = [ctx.interleaved(g.t) for g in feats]
    else:
        parts = [ctx.row_major(g.t) for g in feats]
    out = ttnn.concat(parts, dim=3, memory_config=ctx.mem)
    return Feat(out, n, h, w, sum(g.c for g in feats))


########################################################################################################################################################################
#   modules                                                                                                                                                            #
########################################################################################################################################################################


class TtConv:
    """ultralytics Conv (conv + folded BN + SiLU) or a plain nn.Conv2d, run with ttnn.conv2d.

    out_slice: keep only these output channels (used to split C2f.cv1 into its two halves, see TtC2f)
    pad_in_to: zero-pad the input-channel dim of the weights (the stem conv sees the image padded from 3 to 16 channels)
    in_scale:  fold a constant input scale into the weights (the stem conv takes 0-255 pixels: conv(x/255) == (W/255) conv x)
    """

    def __init__(self, ctx, module, out_slice=None, pad_in_to=None, in_scale=None, activation="auto"):
        conv, w, b = _conv_weights(module)
        if conv.padding_mode != "zeros" or isinstance(conv.padding, str):
            raise NotImplementedError("only zero padding with explicit sizes is ported")
        if in_scale is not None:
            w = w * in_scale
        if out_slice is not None:
            w, b = w[out_slice], b[out_slice]
        if pad_in_to is not None and w.shape[1] < pad_in_to:
            if conv.groups != 1:
                raise NotImplementedError("input-channel padding of grouped convs")
            w = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, pad_in_to - w.shape[1]))

        self.ctx = ctx
        self.groups = conv.groups
        self.out_channels = w.shape[0]
        self.in_channels = w.shape[1] * conv.groups
        self.kernel_size = tuple(conv.kernel_size)
        self.stride = tuple(conv.stride)
        self.padding = tuple(conv.padding)
        self.dilation = tuple(conv.dilation)

        #host tensors; conv2d prepares (tilizes / reorders / casts) them for the card on the first call
        self.weight = ttnn.from_torch(w, dtype=ctx.host_weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.bias = ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ctx.host_weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._prepared = {}  #(batch, h, w) -> device weight/bias returned by conv2d

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ctx.weights_dtype,
            activation=_activation(module) if activation == "auto" else activation,
            deallocate_activation=False,          #inputs are often reused (skip connections), never free them inside the op
            shard_layout=None,                    #let conv2d pick height/block/width sharding for each shape
            output_layout=ttnn.TILE_LAYOUT,
            config_tensors_in_dram=ctx.act_mem == "dram",
        )

    def __call__(self, f: Feat) -> Feat:
        if f.c != self.in_channels:
            raise ValueError(f"conv expects {self.in_channels} input channels, got {f.c}")
        #input in DRAM -> conv2d's DRAM-sliced path (or the matmul path for 1x1); input in L1 -> the L1 path
        x = self.ctx.tile(f.t)
        key = (f.n, f.h, f.w)
        weight, bias = self._prepared.get(key, (self.weight, self.bias))
        out, (out_h, out_w), (weight, bias) = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
            device=self.ctx.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=f.n,
            input_height=f.h,
            input_width=f.w,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.ctx.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self._prepared[key] = (weight, bias)
        out = self.ctx.interleaved(out)
        if out.shape[-1] != self.out_channels:  #some releases return tile-padded channels; drop the padding
            out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, out.shape[-2], self.out_channels], memory_config=self.ctx.mem)
        return Feat(out, f.n, out_h, out_w, self.out_channels)


class TtBottleneck:
    """Bottleneck: cv2(cv1(x)), plus x when shortcut is on."""

    def __init__(self, ctx, m):
        self.ctx = ctx
        self.cv1 = TtConv(ctx, m.cv1)
        self.cv2 = TtConv(ctx, m.cv2)
        self.add = bool(getattr(m, "add", False))

    def __call__(self, f: Feat) -> Feat:
        hidden = self.cv1(f)
        y = self.cv2(hidden)
        ttnn.deallocate(hidden.t)
        if self.add:
            y = Feat(ttnn.add(self.ctx.tile(f.t), y.t, memory_config=self.ctx.mem), y.n, y.h, y.w, y.c)
        return y


class TtC2f:
    """C2f: split cv1 into halves, chain the bottlenecks on the second half, concat everything, cv2.

    cv1(x).chunk(2) is computed as two convs that each produce one half (cv1's weights split along output channels).
    The result is identical and it avoids slicing a tiled tensor at a non-tile-aligned channel (yolov8n's first C2f has
    16-channel halves). tt-metal's own YOLOv8 port does the same.
    """

    def __init__(self, ctx, m):
        self.ctx = ctx
        c = int(m.c)
        self.cv1_a = TtConv(ctx, m.cv1, out_slice=slice(0, c))
        self.cv1_b = TtConv(ctx, m.cv1, out_slice=slice(c, 2 * c))
        self.m = [TtBottleneck(ctx, b) for b in m.m]
        self.cv2 = TtConv(ctx, m.cv2)

    def __call__(self, f: Feat) -> Feat:
        ys = [self.cv1_a(f), self.cv1_b(f)]
        for block in self.m:
            ys.append(block(ys[-1]))
        cat = concat_channels(self.ctx, ys)
        for g in ys:
            ttnn.deallocate(g.t)
        out = self.cv2(cat)
        ttnn.deallocate(cat.t)
        return out


class TtSPPF:
    """SPPF: cv1, n chained 5x5/stride-1 max pools, concat, cv2 (and the optional shortcut newer ultralytics versions add)."""

    def __init__(self, ctx, m):
        self.ctx = ctx
        self.cv1 = TtConv(ctx, m.cv1)
        self.cv2 = TtConv(ctx, m.cv2)
        k = m.m.kernel_size
        self.k = int(k[0] if isinstance(k, (tuple, list)) else k)
        self.n = int(getattr(m, "n", 3))           #pickled v8 models predate these attributes; same defaults as SPPF.forward
        self.add = bool(getattr(m, "add", False))

    def __call__(self, f: Feat) -> Feat:
        y0 = self.cv1(f)
        first = self.ctx.row_major(y0.t)           #max_pool2d consumes row-major [1, 1, N*H*W, C]
        pooled = [first]
        for _ in range(self.n):
            p = ttnn.max_pool2d(
                input_tensor=pooled[-1],
                batch_size=y0.n,
                input_h=y0.h,
                input_w=y0.w,
                channels=y0.c,
                kernel_size=[self.k, self.k],
                stride=[1, 1],
                padding=[self.k // 2, self.k // 2],
                dilation=[1, 1],
                ceil_mode=False,
                memory_config=self.ctx.mem,
            )
            pooled.append(self.ctx.interleaved(p))
        cat = ttnn.concat(pooled, dim=3, memory_config=self.ctx.mem)
        for t in pooled:
            ttnn.deallocate(t)
        if first is not y0.t:
            ttnn.deallocate(y0.t)
        out = self.cv2(Feat(cat, y0.n, y0.h, y0.w, y0.c * (self.n + 1)))
        ttnn.deallocate(cat)
        if self.add:
            out = Feat(ttnn.add(self.ctx.tile(f.t), out.t, memory_config=self.ctx.mem), out.n, out.h, out.w, out.c)
        return out


class TtUpsample:
    """nn.Upsample(scale_factor=2, mode='nearest') via ttnn.upsample on a row-major [N, H, W, C] tensor."""

    def __init__(self, ctx, m):
        self.ctx = ctx
        sf = m.scale_factor
        sf = sf[0] if isinstance(sf, (tuple, list)) else sf
        if m.mode != "nearest" or m.size is not None or sf is None or float(sf) != int(sf):
            raise NotImplementedError("only integer nearest-neighbour upsampling is ported")
        self.scale = int(sf)

    def __call__(self, f: Feat) -> Feat:
        x = ttnn.reshape(self.ctx.row_major(f.t), (f.n, f.h, f.w, f.c))
        x = self.ctx.interleaved(ttnn.upsample(x, self.scale, memory_config=self.ctx.mem))
        h, w = f.h * self.scale, f.w * self.scale
        x = ttnn.reshape(x, (1, 1, f.n * h * w, f.c))
        return Feat(x, f.n, h, w, f.c)


class TtConcat:
    """ultralytics Concat along the channel dim."""

    def __init__(self, ctx, m):
        if int(getattr(m, "d", 1)) != 1:
            raise NotImplementedError("only channel concat (d=1) is ported")
        self.ctx = ctx

    def __call__(self, feats) -> Feat:
        return concat_channels(self.ctx, feats)


def _conv_chain(ctx, seq):
    """Flatten a Detect branch (nn.Sequential of Conv / DWConv / nn.Conv2d, possibly nested) into TtConvs."""
    layers = []
    for sub in seq if isinstance(seq, (nn.Sequential, nn.ModuleList)) else [seq]:
        if isinstance(sub, (nn.Sequential, nn.ModuleList)):
            layers += _conv_chain(ctx, sub)
        elif isinstance(sub, nn.Conv2d):
            layers.append(TtConv(ctx, sub, activation=None))
        elif isinstance(getattr(sub, "conv", None), nn.Conv2d):
            layers.append(TtConv(ctx, sub))
        else:
            raise NotImplementedError(f"{type(sub).__name__} inside the Detect head is not ported")
    return layers


class HeadOutput:
    """What C_head leaves on the card, plus how to bring it back in ultralytics' prediction layout [N, 4 + nc, anchors]."""

    def __init__(self, head, decoded=None, levels=None):
        self.head = head
        self.decoded = decoded   #device decode: [N, anchors, BOX_COLS + nc] row-major (xywh in cols 0-3, class probs after BOX_COLS)
        self.levels = levels     #host decode: per level (box Feat [.., 4*reg_max], cls Feat [.., nc]) raw conv outputs

    @property
    def decoded_on_device(self):
        return self.decoded is not None

    def to_host(self):
        """Device -> host copy (blocks until the card has finished the head)."""
        if self.decoded is not None:
            return ttnn.to_torch(self.decoded).float()
        return [(ttnn.to_torch(b.t).float(), ttnn.to_torch(c.t).float(), b.n, b.h, b.w) for b, c in self.levels]

    def assemble(self, host):
        """Host tensor(s) from to_host() -> [N, 4 + nc, anchors] float32, the format non_max_suppression expects."""
        nc = self.head.nc
        if self.decoded is not None:
            return torch.cat((host[..., :4], host[..., BOX_COLS : BOX_COLS + nc]), -1).transpose(1, 2)
        return self.head.decode_on_host(host)


class TtDetect:
    """Detect head: per level a box branch (cv2: 3 convs -> 4*reg_max) and a class branch (cv3: 3 convs -> nc), then decode.

    On-device decode (default) without softmax over a 16-wide (non-tile-aligned) axis:
        e    = exp(box_logits)                       [R, 64]
        norm = e @ ONES   (per side: sum of 16 bins)  [R, 32]
        num  = e @ BINS   (per side: sum k * e_k)     [R, 32]
        dist = num / norm  = DFL expectation l,t,r,b in grid cells (identical to softmax-then-arange-conv)
        xywh = dist @ DEC * stride + anchor_xy * stride  (dist2bbox in matrix form)
    and class probabilities = sigmoid(cls_logits). Everything stays in tile-aligned matmuls / eltwise ops.
    """

    def __init__(self, ctx, m, decode_on_device=True):
        if getattr(m, "end2end", False):
            raise NotImplementedError("end-to-end (NMS-free) Detect heads are not ported")
        self.ctx = ctx
        self.nc, self.nl, self.reg_max = int(m.nc), int(m.nl), int(m.reg_max)
        if self.reg_max <= 1:
            raise NotImplementedError("Detect heads without DFL (reg_max=1) are not ported")
        self.strides = [float(s) for s in m.stride]
        self.box_branches = [_conv_chain(ctx, m.cv2[i]) for i in range(self.nl)]
        self.cls_branches = [_conv_chain(ctx, m.cv3[i]) for i in range(self.nl)]
        self.decode_on_device = decode_on_device
        self._dfl_mats = None
        self._dec_mats = {}
        self._anchors = {}

    # ------------------------------------------------------------------ constants (built once, kept on the card)
    def _constant(self, x):
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.ctx.device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self.ctx.interleaved(t)

    def _get_dfl_mats(self):
        if self._dfl_mats is None:
            r = self.reg_max
            ones = torch.zeros(4 * r, BOX_COLS)
            bins = torch.zeros(4 * r, BOX_COLS)
            for side in range(4):
                ones[side * r : (side + 1) * r, side] = 1.0
                bins[side * r : (side + 1) * r, side] = torch.arange(r, dtype=torch.float32)
            ones[0, 4:] = 1.0  #padding columns become 0 / e[:, 0] = 0 instead of 0 / 0
            self._dfl_mats = (self._constant(ones.reshape(1, 1, 4 * r, BOX_COLS)),
                              self._constant(bins.reshape(1, 1, 4 * r, BOX_COLS)))
        return self._dfl_mats

    def _get_dec_mat(self, level):
        if level not in self._dec_mats:
            s = self.strides[level]
            dec = torch.zeros(BOX_COLS, BOX_COLS)  #rows: l, t, r, b   cols: cx, cy, w, h (offsets from the anchor)
            dec[0, 0], dec[2, 0] = -0.5, 0.5       #cx = (r - l) / 2
            dec[1, 1], dec[3, 1] = -0.5, 0.5       #cy = (b - t) / 2
            dec[0, 2], dec[2, 2] = 1.0, 1.0        #w  = l + r
            dec[1, 3], dec[3, 3] = 1.0, 1.0        #h  = t + b
            self._dec_mats[level] = self._constant((dec * s).reshape(1, 1, BOX_COLS, BOX_COLS))
        return self._dec_mats[level]

    def _get_anchors(self, level, n, h, w):
        key = (level, n, h, w)
        if key not in self._anchors:
            s = self.strides[level]
            yy, xx = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")  #same as make_anchors
            a = torch.zeros(h * w, BOX_COLS)
            a[:, 0] = xx.reshape(-1) * s
            a[:, 1] = yy.reshape(-1) * s
            self._anchors[key] = self._constant(a.repeat(n, 1).reshape(1, 1, n * h * w, BOX_COLS))
        return self._anchors[key]

    # ------------------------------------------------------------------ forward
    def __call__(self, feats) -> HeadOutput:
        levels = []
        for i, f in enumerate(feats):
            box, cls = f, f
            for layer in self.box_branches[i]:
                box = layer(box)
            for layer in self.cls_branches[i]:
                cls = layer(cls)
            levels.append((box, cls))
        if not self.decode_on_device:
            return HeadOutput(self, levels=levels)

        decoded = [self._decode_level(i, box, cls) for i, (box, cls) in enumerate(levels)]
        out = decoded[0] if len(decoded) == 1 else ttnn.concat(decoded, dim=1, memory_config=self.ctx.mem)
        return HeadOutput(self, decoded=out)

    def _decode_level(self, level, box, cls):
        ctx = self.ctx
        mm = dict(memory_config=ctx.mem, compute_kernel_config=ctx.precise_compute_config)
        ones, bins = self._get_dfl_mats()
        e = ttnn.exp(box.t, memory_config=ctx.mem)
        norm = ttnn.matmul(e, ones, **mm)
        num = ttnn.matmul(e, bins, **mm)
        dist = ttnn.div(num, norm, memory_config=ctx.mem)
        xywh = ttnn.matmul(dist, self._get_dec_mat(level), **mm)
        xywh = ttnn.add(xywh, self._get_anchors(level, box.n, box.h, box.w), memory_config=ctx.mem)
        probs = ttnn.sigmoid(cls.t, memory_config=ctx.mem)
        out = ttnn.concat([ctx.row_major(xywh), ctx.row_major(probs)], dim=3, memory_config=ctx.mem)
        for t in (e, norm, num, dist, xywh, probs):
            ttnn.deallocate(t)
        return ttnn.reshape(out, (box.n, box.h * box.w, BOX_COLS + self.nc))

    def decode_on_host(self, host_levels):
        """fp32 CPU decode of the raw head maps (DFL softmax + dist2bbox + sigmoid), matching ultralytics Detect._inference."""
        boxes, scores, anchors, strides = [], [], [], []
        for level, (b, c, n, h, w) in enumerate(host_levels):
            boxes.append(b.reshape(n, h * w, -1)[..., : 4 * self.reg_max])
            scores.append(c.reshape(n, h * w, -1)[..., : self.nc])
            yy, xx = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
            anchors.append(torch.stack((xx.reshape(-1), yy.reshape(-1)), -1))
            strides.append(torch.full((h * w, 1), self.strides[level]))
        box, score = torch.cat(boxes, 1), torch.cat(scores, 1)
        anchor, stride = torch.cat(anchors), torch.cat(strides)
        n, a, _ = box.shape
        prob = box.view(n, a, 4, self.reg_max).softmax(-1)
        dist = (prob * torch.arange(self.reg_max, dtype=prob.dtype)).sum(-1)
        x1y1 = anchor - dist[..., :2]
        x2y2 = anchor + dist[..., 2:]
        xywh = torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), -1) * stride
        return torch.cat((xywh, score.sigmoid()), -1).transpose(1, 2)


########################################################################################################################################################################
#   model                                                                                                                                                              #
########################################################################################################################################################################

_BUILDERS = {
    "Conv": TtConv,
    "C2f": TtC2f,
    "SPPF": TtSPPF,
    "Upsample": TtUpsample,
    "Concat": TtConcat,
}


class TtYoloV8:
    """TT-NN mirror of a (fused) ultralytics YOLOv8 DetectionModel.

    `model` is the list of TT layers (same order and `i` / `f` as torch_model.model) and `save` is torch_model.save,
    so the CUDA version's run_layers() walks it without changes.
    """

    def __init__(self, torch_model, ctx, decode_on_device=True):
        self.ctx = ctx
        self.save = list(torch_model.save)
        self.names = torch_model.names
        self.model = []
        for m in torch_model.model:
            kind = type(m).__name__
            if kind == "Detect":
                layer = TtDetect(ctx, m, decode_on_device=decode_on_device)
            elif kind == "Conv" and m.i == 0:
                #stem conv sees the 16-channel padded image in 0-255 units; the /255 normalization is folded into its weights
                layer = TtConv(ctx, m, pad_in_to=IMG_CHANNELS_PADDED, in_scale=1.0 / 255.0)
            elif kind in _BUILDERS:
                layer = _BUILDERS[kind](ctx, m)
            else:
                raise NotImplementedError(
                    f"layer {m.i} is {kind}; only YOLOv8-style models ({', '.join(_BUILDERS)}, Detect) are ported"
                )
            layer.i, layer.f, layer.type = m.i, m.f, kind
            self.model.append(layer)
