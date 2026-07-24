"""
ph_loss_stable.py
=================
Stable topological trajectory loss: TRUE H0 persistence + optional PCA
projection — the literature-backed repair of the v2 PH surrogate.

Why (verified refs in reports/deeponet_improvement_research.md):
* The v2 surrogate (ph_loss.py) matches the top-k LARGEST pairwise distances —
  that is a global-spread spectrum, not persistence (H0 Vietoris-Rips deaths
  are the MINIMUM-spanning-tree edge lengths, i.e. connectivity scales).
* Chunks live in padded R^32 (7 real DoF + 25 pad dims): the HDLSS regime
  where distance-based topology is provably noise-dominated
  (arXiv 2404.18194; spectral fix 2311.03087). Fix: project to a few PCA dims
  BEFORE the filtration.

This module provides:
    h0_death_spectrum(x, k)          — differentiable TRUE H0 death spectrum
                                       (batched Prim's MST; validated against
                                       ripser in the self-check).
    ph_h0_loss(pred, target, ...)    — drop-in replacement for
                                       ph_surrogate_loss with proj_dim option.
    PHLossStable(...)                — nn.Module wrapper (same API as PHLoss).

Run `python ph_loss_stable.py` for self-checks (needs ripser for validation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

_INF = 1e10


def h0_death_spectrum(x: Tensor, k: int | None = None) -> Tensor:
    """TRUE 0-dim Vietoris-Rips persistence deaths of each point cloud.

    H0 VR death times are exactly the MST edge lengths (T-1 finite deaths).
    Computed with a batched Prim's algorithm; the returned values are gathered
    from the differentiable distance matrix, so gradients flow to x (indices,
    like topk, are selected without grad — standard subgradient).

    x: (B, T, A) -> (B, k or T-1) death spectrum sorted descending.
    """
    if x.dim() != 3:
        raise ValueError(f"expected (B,T,A), got {tuple(x.shape)}")
    B, T, _ = x.shape
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    d = torch.cdist(x, x)                                    # (B,T,T)
    bidx = torch.arange(B, device=x.device)

    intree = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    intree[:, 0] = True                                      # pre-graph init ok
    dist = d[:, 0, :].masked_fill(intree, _INF)
    deaths = []
    for _ in range(T - 1):
        j = dist.argmin(dim=1)                               # (B,)
        deaths.append(dist.gather(1, j.unsqueeze(1)).squeeze(1))
        # out-of-place mask update: autograd saves masks, never mutate them
        intree = intree.scatter(1, j.unsqueeze(1), True)
        dist = torch.minimum(dist, d[bidx, j, :]).masked_fill(intree, _INF)
    spec = torch.stack(deaths, dim=1)                        # (B, T-1)
    spec, _ = torch.sort(spec, dim=1, descending=True)
    if k is not None:
        spec = spec[:, : min(k, spec.shape[1])]
    return spec


def _pca_project(pred: Tensor, target: Tensor, d: int):
    """Project both chunks onto the target's top-d principal directions.

    Basis is computed from the TARGET only (detached), so the projection is a
    fixed linear map w.r.t. pred -> gradients flow through pred cleanly.
    """
    with torch.no_grad():
        mu = target.mean(dim=1, keepdim=True)                # (B,1,A)
        _, _, Vh = torch.linalg.svd(target - mu, full_matrices=False)
        P = Vh[:, :d, :]                                     # (B,d,A)
    pred_p = torch.einsum("bta,bda->btd", pred - mu, P)
    targ_p = torch.einsum("bta,bda->btd", target - mu, P)
    return pred_p, targ_p


def ph_h0_loss(pred: Tensor, target: Tensor, k: int = 8,
               proj_dim: int | None = 3, reduction: str = "mean") -> Tensor:
    """Stable topological loss: L1 between TRUE H0 death spectra, computed in
    a low-dimensional PCA subspace of the target chunk (proj_dim=None disables
    projection). Same API/reductions as ph_surrogate_loss.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
    # SVD/cdist lack bf16 CPU kernels; the cast is differentiable (as in ph_loss.py)
    if pred.dtype not in (torch.float32, torch.float64):
        pred, target = pred.float(), target.float()
    if proj_dim is not None and proj_dim < pred.shape[-1]:
        pred, target = _pca_project(pred, target, proj_dim)
    pred_spec = h0_death_spectrum(pred, k=k)
    with torch.no_grad():
        targ_spec = h0_death_spectrum(target, k=k)
    per_sample = (pred_spec - targ_spec).abs().mean(dim=1)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction '{reduction}'")


class PHLossStable(nn.Module):
    """Drop-in for PHLoss: PHLossStable(k=8, proj_dim=3)."""

    def __init__(self, k: int = 8, proj_dim: int | None = 3, reduction: str = "mean"):
        super().__init__()
        self.k, self.proj_dim, self.reduction = k, proj_dim, reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return ph_h0_loss(pred, target, k=self.k, proj_dim=self.proj_dim,
                          reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"k={self.k}, proj_dim={self.proj_dim}, reduction={self.reduction}"


# ---------------------------------------------------------------- self-check
def _run_tests():
    torch.manual_seed(0)
    B, T, A = 4, 16, 7

    # 1. our MST spectrum == ripser's H0 deaths (ground truth)
    import numpy as np
    from ripser import ripser
    x = torch.randn(B, T, A)
    ours = h0_death_spectrum(x)                              # (B, T-1)
    for b in range(B):
        dgm = ripser(x[b].numpy(), maxdim=0)["dgms"][0]
        deaths = np.sort(dgm[np.isfinite(dgm[:, 1]), 1])[::-1]
        assert np.allclose(ours[b].numpy(), deaths, atol=1e-4), \
            f"MST spectrum != ripser H0 deaths (sample {b})"
    print("[1] h0_death_spectrum == ripser H0 deaths          OK")

    # 2. identical -> 0; different -> >0
    assert ph_h0_loss(x, x.clone()).item() < 1e-6
    assert ph_h0_loss(x, torch.randn(B, T, A)).item() > 0
    print("[2] zero on identical, positive on different       OK")

    # 3. grads flow, finite
    p = torch.randn(B, T, A, requires_grad=True)
    l = ph_h0_loss(p, torch.randn(B, T, A))
    l.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all() \
        and p.grad.abs().sum() > 0
    print("[3] gradient finite and nonzero                     OK")

    # 4. optimization reduces the loss
    p2 = torch.randn(B, T, A, requires_grad=True)
    t2 = torch.randn(B, T, A)
    opt = torch.optim.SGD([p2], lr=0.05)
    l0 = ph_h0_loss(p2, t2).item()
    for _ in range(200):
        opt.zero_grad(); l = ph_h0_loss(p2, t2); l.backward(); opt.step()
    l1 = ph_h0_loss(p2, t2).item()
    assert l1 < l0
    print(f"[4] optimize: {l0:.4f} -> {l1:.4f} (decreasing)        OK")

    # 5. projection path + bf16 stay finite
    xb = x.to(torch.bfloat16)
    lb = ph_h0_loss(xb, torch.randn(B, T, A, dtype=torch.bfloat16))
    assert torch.isfinite(lb)
    l32 = ph_h0_loss(torch.randn(2, 50, 32), torch.randn(2, 50, 32), proj_dim=3)
    assert torch.isfinite(l32)
    print("[5] PCA projection + bf16 finite                    OK")
    print("\n[ph_loss_stable] ALL TESTS PASSED")


if __name__ == "__main__":
    _run_tests()
