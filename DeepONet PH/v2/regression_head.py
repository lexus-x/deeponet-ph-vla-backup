"""
regression_head.py
==================
Control baseline for the "is it DeepONet specifically?" ablation.

RegressionHeadV2 keeps the EXACT same cross-attention context encoder as
DeepONet-v2 (CrossAttnPool: same n_queries, n_blocks, d_model), but replaces the
branch(obs) (x) trunk(tau) operator with a plain MLP that maps the pooled context
directly to the whole flattened action chunk. So the ONLY difference vs
DeepONet-v2 is the operator structure -> if this matches DeepONet-v2, the win is
"single-pass + cross-attention", not the operator; if DeepONet-v2 wins, the
operator factorization matters. Params are matched to ~10.4M.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from deeponet_head_v2 import CrossAttnPool, _mlp


class RegressionHeadV2(nn.Module):
    def __init__(self, context_dim, chunk_size, action_dim, d_model=512, n_queries=8,
                 n_heads=8, n_blocks=3, hidden=640):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.pool = CrossAttnPool(context_dim, d_model, n_queries, n_heads, n_blocks)
        # plain MLP: flattened context -> entire action chunk (no tau/operator structure)
        self.decoder = _mlp([n_queries * d_model, hidden, chunk_size * action_dim],
                            act=nn.GELU, layernorm=True)

    def forward(self, prefix: Tensor, pad_mask: Tensor) -> Tensor:
        ctx = self.pool(prefix, pad_mask)                 # (B, K, d_model)
        out = self.decoder(ctx.flatten(1))                # (B, T*A)
        return out.view(-1, self.chunk_size, self.action_dim)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    B, N, D, T, A = 2, 600, 960, 50, 32
    h = RegressionHeadV2(context_dim=D, chunk_size=T, action_dim=A)
    x = torch.randn(B, N, D); m = torch.ones(B, N, dtype=torch.bool)
    y = h(x, m); assert y.shape == (B, T, A), y.shape
    print(f"[reg head] out {tuple(y.shape)}  params={h.num_params()/1e6:.2f}M")
    y.sum().backward(); print("[reg head] backward OK")
