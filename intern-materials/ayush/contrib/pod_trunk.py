"""
pod_trunk.py
============
POD "eigen-motion" trunk for the DeepONet action head (POD-DeepONet applied to
robot action chunks — Lu et al., CMAME 2022, arXiv 2111.05512, sec. on POD trunks).

Idea: demo action chunks a_i(tau) in R^A are smooth and low-rank. Compute the
POD/PCA basis of the training chunks once; the k-th mode psi_k: [0,1] -> R^A is
an interpretable "eigen-motion". The head then only learns the branch:

    a(tau) = mean(tau) + sum_k  c_k(obs) * psi_k(tau)

This replaces trunk + out_mlp with a FIXED orthonormal, provably
well-conditioned basis (two-step-training argument of Lee & Shin, 2309.01020,
comes for free: the basis is orthonormal by construction). Continuous-tau
queries use linear interpolation of the basis (chunks are dense in tau, so
interpolation error is negligible next to model error).

Drop-in: PODHead reuses CrossAttnPool from deeponet_head_v2.py unchanged.
Run `python pod_trunk.py` for the self-check.
"""

from __future__ import annotations

import sys, os
import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "git-repo", "DeepONet PH", "v2"))


def fit_pod(chunks: Tensor, p: int):
    """POD basis of demo chunks.

    chunks: (N, T, A) training action chunks.
    Returns (mean (T,A), basis (p,T,A), explained (p,), sing_values (min(N,TA),)).
    Basis rows are orthonormal in the flattened (T*A) inner product.
    """
    N, T, A = chunks.shape
    X = chunks.reshape(N, T * A)
    mean = X.mean(0)
    Xc = X - mean
    # SVD of the centered snapshot matrix
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    explained = (var / var.sum())[:p]
    basis = Vh[:p]                                   # (p, T*A), orthonormal
    return mean.reshape(T, A), basis.reshape(p, T, A), explained, S


class PODHead(nn.Module):
    """DeepONet head with a fixed POD trunk: branch coefficients x eigen-motions.

    Same (prefix, pad_mask) -> (B, T, A) interface as DeepONetHeadV2.
    """

    def __init__(self, context_dim, mean: Tensor, basis: Tensor,
                 d_model=512, n_queries=8, n_heads=8, n_blocks=3,
                 branch_hidden=768, coeff_scale=None):
        super().__init__()
        from deeponet_head_v2 import CrossAttnPool, _mlp
        p, T, A = basis.shape
        self.p, self.chunk_size, self.action_dim = p, T, A
        self.pool = CrossAttnPool(context_dim, d_model, n_queries, n_heads, n_blocks)
        self.branch = _mlp([n_queries * d_model, branch_hidden, p],
                           act=nn.GELU, layernorm=True)
        self.register_buffer("mean", mean)              # (T, A)
        self.register_buffer("basis", basis)            # (p, T, A)
        # scale branch outputs to the data's coefficient magnitudes so the
        # branch works near unit scale (helps optimization)
        if coeff_scale is None:
            coeff_scale = torch.ones(p)
        self.register_buffer("coeff_scale", coeff_scale)  # (p,)

    def forward(self, prefix: Tensor, pad_mask: Tensor) -> Tensor:
        ctx = self.pool(prefix, pad_mask)               # (B,K,d)
        c = self.branch(ctx.flatten(1)) * self.coeff_scale  # (B,p)
        a = torch.einsum("bp,pta->bta", c, self.basis)  # (B,T,A)
        return a + self.mean

    def query(self, c: Tensor, tau: Tensor) -> Tensor:
        """Continuous-tau query by linear interpolation of mean+basis.

        c (B,p) UNSCALED branch output; tau (Q,) in [0,1] -> (B,Q,A).
        """
        T = self.chunk_size
        pos = tau.clamp(0, 1) * (T - 1)
        i0 = pos.floor().long().clamp(max=T - 2)
        w = (pos - i0.to(pos.dtype)).view(1, -1, 1)
        def interp(f):  # f (..., T, A) -> (..., Q, A)
            return f[..., i0, :] * (1 - w) + f[..., i0 + 1, :] * w
        basis_q = interp(self.basis.unsqueeze(0))[0]     # (p,Q,A) -> broadcast ok
        mean_q = interp(self.mean.unsqueeze(0))          # (1,Q,A)
        a = torch.einsum("bp,pqa->bqa", c * self.coeff_scale, basis_q)
        return a + mean_q

    def num_params(self):
        return sum(q.numel() for q in self.parameters())


def coeff_stats(chunks: Tensor, mean: Tensor, basis: Tensor):
    """Project demo chunks onto the basis -> per-mode coefficient std (p,)."""
    N, T, A = chunks.shape
    Xc = chunks.reshape(N, T * A) - mean.reshape(-1)
    C = Xc @ basis.reshape(basis.shape[0], -1).T          # (N, p)
    return C.std(0)


# ---------------------------------------------------------------- self-check
def _self_check():
    torch.manual_seed(0)
    N, T, A, p = 400, 50, 32, 16
    # synthetic smooth low-rank chunks: random combos of 6 smooth modes
    tgrid = torch.linspace(0, 1, T)
    true_modes = torch.stack([torch.sin((k + 1) * torch.pi * tgrid)
                              for k in range(6)])            # (6, T)
    W = torch.randn(N, 6)
    M = torch.randn(6, A) * 0.5
    chunks = torch.einsum("nk,kt,ka->nta", W, true_modes, M) \
        + 0.01 * torch.randn(N, T, A)

    mean, basis, expl, S = fit_pod(chunks, p)
    assert expl[:6].sum() > 0.98, f"6 true modes should explain >98%, got {expl[:6].sum():.3f}"
    print(f"[1] POD recovers low-rank structure: top-6 explain {expl[:6].sum()*100:.1f}%  OK")

    # orthonormality
    Bf = basis.reshape(p, -1)
    G = Bf @ Bf.T
    assert torch.allclose(G, torch.eye(p), atol=1e-4), "basis not orthonormal"
    print("[2] basis orthonormal (Gram = I)                    OK")

    # head forward + param count
    scale = coeff_stats(chunks, mean, basis)
    head = PODHead(context_dim=960, mean=mean, basis=basis, coeff_scale=scale)
    prefix = torch.randn(2, 64, 960); mask = torch.ones(2, 64, dtype=torch.bool)
    y = head(prefix, mask)
    assert y.shape == (2, T, A)
    print(f"[3] forward (2,{T},{A}) OK; params {head.num_params()/1e6:.2f}M "
          f"(vs ~10.4M v2 head)")

    # continuous query at grid == forward
    ctx = head.pool(prefix, mask)
    c = head.branch(ctx.flatten(1))
    yq = head.query(c, tgrid)
    assert torch.allclose(yq, y, atol=1e-4), "query(grid) != forward"
    print("[4] continuous query(grid) == forward               OK")

    y.sum().backward()
    print("[5] backward OK\n\n[pod_trunk] ALL CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
