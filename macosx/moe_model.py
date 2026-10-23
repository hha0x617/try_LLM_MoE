"""
Mixtral-compatible MoE (Mixture of Experts) transformer. (macOS version of ../moe_model.py)

Difference from ../moe_model.py:
  - RoPE uses real-valued cos/sin instead of complex tensors (complex ops are not fully supported on MPS).
    The result is mathematically identical, and the rope buffer is not saved in checkpoints,
    so checkpoints are interchangeable between the two versions.

Tensor shapes and formulas follow the llama.cpp implementation so the model can be written
directly to the GGUF "llama" architecture (treated as a Mixtral-style MoE when expert_count > 0).

  - RMSNorm
  - RoPE: rotates adjacent pairs (x[2i], x[2i+1]) (llama.cpp's LLAMA_ROPE_TYPE_NORM)
          Note: differs from HF transformers' rotate_half, so no Q/K permutation is needed for GGUF conversion
  - GQA (Grouped Query Attention)
  - MoE FFN: softmax router -> Top-k -> renormalize the k selected weights -> SwiGLU experts

Expert weights are held as 3-D tensors with all experts stacked into one.
  w_gate, w_up : [n_expert, d_ff, d_model]   -> GGUF: blk.N.ffn_gate_exps / ffn_up_exps
  w_down       : [n_expert, d_model, d_ff]   -> GGUF: blk.N.ffn_down_exps
  router       : [n_expert, d_model]         -> GGUF: blk.N.ffn_gate_inp
"""
import math
from dataclasses import dataclass, asdict, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    vocab_size: int = 32000
    block_size: int = 1024      # context length
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 4          # number of GQA KV heads (must divide n_head)
    d_model: int = 512
    d_ff: int = 1408            # hidden dimension per expert
    n_expert: int = 8
    n_expert_used: int = 2      # k of Top-k
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False
    aux_loss_coef: float = 0.01  # load-balancing loss coefficient

    @property
    def head_dim(self):
        return self.d_model // self.n_head

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# Presets by purpose (total parameter counts are approximate for vocab_size=32000)
PRESETS = {
    # smoke test (finishes in a few minutes)
    "tiny":  dict(block_size=256,  n_layer=2,  n_head=4,  n_kv_head=2, d_model=128, d_ff=256,  n_expert=4, n_expert_used=2),
    # total ~180M / active ~75M : default for RTX 4070 Ti SUPER 16GB
    "small": dict(block_size=1024, n_layer=8,  n_head=8,  n_kv_head=4, d_model=512, d_ff=1408, n_expert=8, n_expert_used=2),
    # total ~520M / active ~180M : on 16GB, lower --batch_size to 2-4 and raise --grad_accum
    "base":  dict(block_size=1024, n_layer=12, n_head=12, n_kv_head=4, d_model=768, d_ff=2048, n_expert=8, n_expert_used=2),
}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def precompute_rope(head_dim, max_len, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)                                # [T, head_dim/2]
    return torch.stack((angles.cos(), angles.sin()), dim=-1)      # [T, head_dim/2, 2] (cos, sin)


def apply_rope(x, rope):
    # x: [B, T, H, D]  rotate each adjacent pair (x[2i], x[2i+1]) by its angle
    B, T, H, D = x.shape
    x0, x1 = x.float().reshape(B, T, H, D // 2, 2).unbind(-1)
    cos, sin = rope[:T].view(1, T, 1, D // 2, 2).unbind(-1)
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.reshape(B, T, H, D).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0 and cfg.n_head % cfg.n_kv_head == 0
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, cfg.d_model, bias=False)

    def forward(self, x, rope):
        B, T, _ = x.shape
        q = apply_rope(self.wq(x).view(B, T, self.n_head, self.head_dim), rope)
        k = apply_rope(self.wk(x).view(B, T, self.n_kv_head, self.head_dim), rope)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, D]
        # GQA: each KV head is shared by a group of consecutive query heads (same mapping as llama.cpp)
        rep = self.n_head // self.n_kv_head
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class MoEFeedForward(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        E, d, f = cfg.n_expert, cfg.d_model, cfg.d_ff
        self.n_expert, self.top_k = E, cfg.n_expert_used
        self.router = nn.Linear(d, E, bias=False)
        # experts stacked into 3-D tensors ([out, in] order, same as nn.Linear)
        self.w_gate = nn.Parameter(torch.empty(E, f, d))
        self.w_up = nn.Parameter(torch.empty(E, f, d))
        self.w_down = nn.Parameter(torch.empty(E, d, f))

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        probs = self.router(xf).float().softmax(dim=-1)        # [N, E]
        top_w, top_i = probs.topk(self.top_k, dim=-1)           # [N, k]
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)         # renormalize, as in Mixtral

        out = torch.zeros_like(xf, dtype=torch.float32)
        for e in range(self.n_expert):
            tok, slot = torch.where(top_i == e)                 # tokens routed to this expert
            h = xf[tok]                                         # Gather
            h = F.silu(h @ self.w_gate[e].t()) * (h @ self.w_up[e].t())
            h = h @ self.w_down[e].t()
            out.index_add_(0, tok, h.float() * top_w[tok, slot, None])  # Scatter (weighted add)

        # Switch Transformer-style load-balancing loss: E * Σ (assignment fraction × mean probability)
        counts = F.one_hot(top_i, self.n_expert).sum(dim=(0, 1)).float()
        frac = counts / (xf.shape[0] * self.top_k)
        aux = self.n_expert * (frac * probs.mean(dim=0)).sum()
        return out.type_as(x).view(B, T, C), aux, counts.detach()


class Block(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.moe = MoEFeedForward(cfg)

    def forward(self, x, rope):
        x = x + self.attn(self.attn_norm(x), rope)
        m, aux, counts = self.moe(self.ffn_norm(x))
        return x + m, aux, counts


class MoETransformer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight
        self.register_buffer("rope", precompute_rope(cfg.head_dim, cfg.block_size, cfg.rope_theta), persistent=False)
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        proj_std = std / math.sqrt(2 * self.cfg.n_layer)  # smaller init for projections added into the residual
        for name, p in self.named_parameters():
            if p.dim() == 1:
                continue  # RMSNorm weights stay at 1
            nn.init.normal_(p, mean=0.0, std=proj_std if name.endswith(("wo.weight", "w_down")) else std)

    def num_params(self):
        cfg = self.cfg
        total = sum(p.numel() for p in self.parameters())
        expert = sum(p.numel() for n, p in self.named_parameters() if n.endswith(("w_gate", "w_up", "w_down")))
        active = total - expert + expert * cfg.n_expert_used // cfg.n_expert
        return total, active

    def forward(self, idx, targets=None):
        x = self.tok_embeddings(idx)
        aux_total, counts = 0.0, []
        for layer in self.layers:
            x, aux, c = layer(x, self.rope)
            aux_total = aux_total + aux
            counts.append(c)
        logits = self.output(self.norm(x))
        loss = ce = None
        aux_loss = aux_total / len(self.layers)
        if targets is not None:
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
            loss = ce + self.cfg.aux_loss_coef * aux_loss
        return {"logits": logits, "loss": loss, "ce": ce, "aux": aux_loss, "counts": torch.stack(counts)}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eos_id=None):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.block_size:])["logits"][:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)  # greedy
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return idx


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = MoEConfig.from_dict(ckpt["config"])
    model = MoETransformer(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt
