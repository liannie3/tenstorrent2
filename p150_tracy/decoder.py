"""Batch-one, eager TT-NN implementation of the stock TinyLlama Llama decoder."""
from types import SimpleNamespace

import torch
import ttnn as tt


class Decoder:
    def __init__(self, reference, device):
        self.device = device
        self.config = cfg = reference.config
        self.generation_config = reference.generation_config
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
        if (cfg.model_type != "llama" or cfg.hidden_act != "silu" or rope.get("rope_type", "default") != "default"
                or getattr(cfg, "attention_bias", False) or getattr(cfg, "mlp_bias", False)):
            raise ValueError("This backend requires stock TinyLlama with unscaled RoPE and bias-free projections.")
        self.embedding = reference.get_input_embeddings()
        self.nq = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.dim = cfg.hidden_size // self.nq
        self.eps = cfg.rms_norm_eps
        self.inverse_freq = float(rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))) ** (-torch.arange(0, self.dim, 2).float() / self.dim)
        self.layers = []
        for layer in reference.model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            self.layers.append(SimpleNamespace(
                norm=self.norm_weight(layer.input_layernorm),
                q=self.matrix(attn.q_proj), k=self.matrix(attn.k_proj),
                v=self.matrix(attn.v_proj), o=self.matrix(attn.o_proj),
                post=self.norm_weight(layer.post_attention_layernorm),
                gate=self.matrix(mlp.gate_proj), up=self.matrix(mlp.up_proj),
                down=self.matrix(mlp.down_proj)))
        self.final_norm = self.norm_weight(reference.model.norm)
        self.output = self.matrix(reference.lm_head)
        tt.synchronize_device(device)

    def upload(self, x):
        return tt.from_torch(x.detach().to(torch.bfloat16).contiguous(), dtype=tt.bfloat16,
                             layout=tt.TILE_LAYOUT, device=self.device, memory_config=tt.DRAM_MEMORY_CONFIG)

    def norm_weight(self, module):
        return self.upload(module.weight.reshape(1, -1))

    def matrix(self, module):
        return self.upload(module.weight.T)

    def get_input_embeddings(self):
        return self.embedding

    def heads(self, x, count, length):
        return tt.permute(tt.reshape(x, (1, length, count, self.dim)), (0, 2, 1, 3))

    def rotate(self, x, cosine, sine):
        half = self.dim // 2
        rotated = tt.concat([tt.neg(x[:, :, :, half:]), x[:, :, :, :half]], dim=3)
        return tt.add(tt.multiply(x, cosine), tt.multiply(rotated, sine))

    def forward_hidden(self, embeddings, positions, cache, keep_cache=True):
        length = embeddings.shape[1]
        past = 0 if cache is None else int(cache[0][0].shape[2])
        if positions is None:
            positions = torch.arange(past, past + length).reshape(1, length)
        phase = positions.float().reshape(-1, 1) * self.inverse_freq
        phase = torch.cat((phase, phase), dim=-1).reshape(1, 1, length, self.dim)
        cosine, sine = self.upload(phase.cos()), self.upload(phase.sin())
        mask = None
        if length > 1:
            blocked = torch.arange(past + length)[None, :] > positions.reshape(-1, 1)
            mask = self.upload(torch.zeros_like(blocked, dtype=torch.float32)
                               .masked_fill(blocked, -1e9).reshape(1, 1, length, past + length))
        x = self.upload(embeddings.reshape(1, 1, length, -1))
        updated = []
        for index, w in enumerate(self.layers):
            h = tt.rms_norm(x, epsilon=self.eps, weight=w.norm)
            q = self.rotate(self.heads(tt.matmul(h, w.q), self.nq, length), cosine, sine)
            k = self.rotate(self.heads(tt.matmul(h, w.k), self.nkv, length), cosine, sine)
            v = self.heads(tt.matmul(h, w.v), self.nkv, length)
            if cache is not None:
                k = tt.concat([cache[index][0], k], dim=2)
                v = tt.concat([cache[index][1], v], dim=2)
            if keep_cache:
                updated.append((k, v))
            repeats = self.nq // self.nkv
            keys = tt.repeat_interleave(k, repeats, dim=1)
            values = tt.repeat_interleave(v, repeats, dim=1)
            scores = tt.multiply(tt.matmul(q, tt.transpose(keys, -1, -2)), self.dim ** -0.5)
            if mask is not None:
                scores = tt.add(scores, mask)
            attention = tt.matmul(tt.softmax(scores, dim=-1), values)
            merged = tt.reshape(tt.permute(attention, (0, 2, 1, 3)), (1, 1, length, -1))
            x = tt.add(x, tt.matmul(merged, w.o))
            h = tt.rms_norm(x, epsilon=self.eps, weight=w.post)
            gated = tt.multiply(tt.silu(tt.matmul(h, w.gate)), tt.matmul(h, w.up))
            x = tt.add(x, tt.matmul(gated, w.down))
        return tt.rms_norm(x, epsilon=self.eps, weight=self.final_norm), updated

    def __call__(self, *, input_ids=None, inputs_embeds=None, position_ids=None,
                 past_key_values=None, **kwargs):
        embeddings = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        h, cache = self.forward_hidden(embeddings, position_ids, past_key_values)
        logits = tt.matmul(h[:, :, -1:, :], self.output)
        return SimpleNamespace(logits=tt.to_torch(logits).reshape(1, 1, -1).float(), past_key_values=cache)

    def model(self, *, input_ids, **kwargs):
        h, _ = self.forward_hidden(self.embedding(input_ids), None, None, keep_cache=False)
        return SimpleNamespace(last_hidden_state=tt.to_torch(h).reshape(1, input_ids.shape[1], -1))
