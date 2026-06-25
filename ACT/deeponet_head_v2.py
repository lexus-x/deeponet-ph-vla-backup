"""
deeponet_head_v2.py
===================
DeepONet action head v2 — same operator-learning idea, but the BRANCH now reads
the FULL observation token sequence via cross-attention instead of a single
mean-pooled vector. This fixes the information bottleneck that made v1 collapse on
spatially-specific tasks (e.g. "bowl on the stove / cabinet").

Components
----------
* CrossAttnPool: K learned query tokens cross-attend to all prefix tokens
  (Perceiver-style), producing K focused context vectors that preserve spatial
  detail. This IS the DeepONet branch's encoder of the input function.
* Branch: maps the K context vectors -> p branch coefficients.
* Trunk: maps query time tau in [0,1] (with Fourier features) -> p basis values.
* Merge (parameter-free) c (x) phi, then a small output MLP -> action chunk.

Still a DeepONet: a(tau) = OutMLP( branch(obs) (x) trunk(tau) ). Target ~10-15M.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _mlp(sizes, act=nn.GELU, layernorm=False, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last or last_act:
            if layernorm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


class CrossBlock(nn.Module):
    """Pre-norm cross-attention block: queries attend to context tokens."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Linear(2 * d_model, d_model))

    def forward(self, q, kv, key_padding_mask):
        qn, kvn = self.ln_q(q), self.ln_kv(kv)
        a, _ = self.attn(qn, kvn, kvn, key_padding_mask=key_padding_mask, need_weights=False)
        q = q + a
        q = q + self.ff(self.ln2(q))
        return q


class CrossAttnPool(nn.Module):
    """K learned queries cross-attend to the full prefix token sequence."""

    def __init__(self, d_in, d_model=512, n_queries=8, n_heads=8, n_blocks=3):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.blocks = nn.ModuleList([CrossBlock(d_model, n_heads) for _ in range(n_blocks)])
        self.n_queries = n_queries
        self.d_model = d_model

    def forward(self, prefix: Tensor, pad_mask: Tensor) -> Tensor:
        # prefix (B,N,d_in); pad_mask (B,N) bool, True=valid token
        B = prefix.shape[0]
        kv = self.in_proj(prefix)                         # (B,N,d_model)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B,K,d_model)
        key_padding = ~pad_mask.bool()                    # True = ignore
        for blk in self.blocks:
            q = blk(q, kv, key_padding)
        return q                                          # (B,K,d_model)


class DeepONetHeadV2(nn.Module):
    def __init__(self, context_dim, chunk_size, action_dim,
                 p=256, d_model=512, n_queries=8, n_heads=8, n_blocks=3,
                 branch_hidden=768, trunk_hidden=256, out_hidden=256, n_fourier=6):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.p = p
        self.n_fourier = n_fourier

        self.pool = CrossAttnPool(context_dim, d_model, n_queries, n_heads, n_blocks)
        self.branch = _mlp([n_queries * d_model, branch_hidden, p], act=nn.GELU, layernorm=True)
        trunk_in = 1 + 2 * n_fourier
        self.trunk = _mlp([trunk_in, trunk_hidden, trunk_hidden, p], act=nn.GELU, layernorm=False)
        self.out_mlp = _mlp([p, out_hidden, action_dim], act=nn.GELU, layernorm=False)
        self.out_bias = nn.Parameter(torch.zeros(action_dim))

        tau = torch.linspace(0.0, 1.0, chunk_size).unsqueeze(-1)  # (T,1)
        self.register_buffer("tau", tau, persistent=False)
        freqs = (2.0 ** torch.arange(n_fourier)) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)

    def _fourier(self, tau):
        # tau (T,1) -> (T, 1+2F)
        ang = tau * self.freqs.to(tau.dtype)              # (T,F)
        return torch.cat([tau, torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, prefix: Tensor, pad_mask: Tensor) -> Tensor:
        ctx = self.pool(prefix, pad_mask)                 # (B,K,d_model)
        c = self.branch(ctx.flatten(1))                   # (B,p)
        phi = self.trunk(self._fourier(self.tau.to(prefix.dtype)))  # (T,p)
        feat = c.unsqueeze(1) * phi.unsqueeze(0)          # (B,T,p)
        return self.out_mlp(feat) + self.out_bias         # (B,T,A)

    def num_params(self):
        return sum(q.numel() for q in self.parameters())


if __name__ == "__main__":
    B, N, D, T, A = 2, 600, 960, 50, 32
    head = DeepONetHeadV2(context_dim=D, chunk_size=T, action_dim=A)
    prefix = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool); mask[0, 400:] = False
    y = head(prefix, mask)
    assert y.shape == (B, T, A), y.shape
    print(f"[v2 head] out {tuple(y.shape)}  params={head.num_params()/1e6:.2f}M")
    y.sum().backward(); print("[v2 head] backward OK")
