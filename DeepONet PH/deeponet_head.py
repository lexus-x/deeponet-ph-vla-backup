"""
deeponet_head.py
================
A DeepONet action head for SmolVLA.

Idea
----
We replace SmolVLA's ~98M iterative flow-matching "action expert" with a small
(~2M) operator-learning head. A DeepONet learns an *operator*: it maps an input
function (here: the multimodal observation, summarised by the VLM backbone into a
context vector) to an output function (here: the action trajectory a(tau) over a
normalised chunk-time tau in [0, 1]).

Architecture (merged DeepONet with output MLP)
----------------------------------------------
    context (B, D)  --Branch MLP-->  c   (B, p)            # "what to do"
    tau     (T, 1)  --Trunk  MLP-->  phi (T, p)            # "trajectory basis"
    feat = c[:, None, :] * phi[None, :, :]   -> (B, T, p)  # Hadamard merge
    a    = OutMLP(feat)                      -> (B, T, A)  # per-step action

The trunk is queried at the T = chunk_size fixed, evenly-spaced points
tau_t = t / (T - 1). Because the trunk is a *continuous* function of tau, the
head is resolution-free: it could be queried at any horizon, not just T.

Key contrasts with flow matching
--------------------------------
* Single forward pass (no iterative denoising) -> lower latency.
* Direct regression to the action chunk (MSE), not a velocity field.
* Tiny parameter count (~2M vs ~98M expert).

This module is the action head only; the SmolVLA wrapper
(modeling_smolvla_deeponet.py) is responsible for producing `context` from the
frozen/unfrozen VLM backbone and for wiring the head into training + inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _mlp(sizes, act=nn.GELU, layernorm=False, last_act=False):
    """Build an MLP from a list of layer sizes."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        if not is_last or last_act:
            if layernorm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


class DeepONetHead(nn.Module):
    """
    DeepONet action head.

    Args
    ----
    context_dim : dim of the pooled VLM context vector (branch input).
    chunk_size  : action horizon T (number of trunk query points).
    action_dim  : output action dimension A (padded max_action_dim, e.g. 32).
    p           : number of basis functions (branch/trunk width at the merge).
    branch_hidden / trunk_hidden / out_hidden : MLP widths.
    """

    def __init__(
        self,
        context_dim: int,
        chunk_size: int,
        action_dim: int,
        p: int = 64,
        branch_hidden: int = 1024,
        trunk_hidden: int = 256,
        out_hidden: int = 256,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.p = p

        # Branch: context -> coefficients c (B, p). LayerNorm for stable training.
        self.branch = _mlp([context_dim, branch_hidden, branch_hidden, p],
                            act=nn.GELU, layernorm=True)
        # Trunk: tau (.,1) -> basis phi (.,p). No LayerNorm (1-D coordinate input).
        self.trunk = _mlp([1, trunk_hidden, trunk_hidden, trunk_hidden, p],
                          act=nn.GELU, layernorm=False)
        # Output MLP: merged feature (.,p) -> action (.,A).
        self.out_mlp = _mlp([p, out_hidden, action_dim], act=nn.GELU, layernorm=False)
        # learnable per-dim output bias (the DeepONet "b_0" term)
        self.out_bias = nn.Parameter(torch.zeros(action_dim))

        # Fixed evenly-spaced trunk query points in [0, 1]; (T, 1) buffer.
        tau = torch.linspace(0.0, 1.0, chunk_size).unsqueeze(-1)
        self.register_buffer("tau", tau, persistent=False)

    def forward(self, context: Tensor) -> Tensor:
        """
        context : (B, context_dim) pooled VLM features.
        returns : (B, T, action_dim) predicted action chunk.
        """
        c = self.branch(context)                     # (B, p)
        phi = self.trunk(self.tau.to(context.dtype))  # (T, p)
        feat = c.unsqueeze(1) * phi.unsqueeze(0)      # (B, T, p)
        out = self.out_mlp(feat) + self.out_bias      # (B, T, A)
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Self-check: shapes + param count.
    B, D, T, A = 4, 960, 50, 32
    head = DeepONetHead(context_dim=D, chunk_size=T, action_dim=A, p=64)
    x = torch.randn(B, D)
    y = head(x)
    assert y.shape == (B, T, A), y.shape
    print(f"[deeponet_head] out shape {tuple(y.shape)}  params={head.num_params()/1e6:.3f}M")
    # gradient sanity
    y.sum().backward()
    print("[deeponet_head] backward OK")
