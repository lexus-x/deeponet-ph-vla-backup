"""
ph_loss.py
==========
Fast, differentiable, batched surrogate for a Persistent-Homology (PH)
regularizer over action chunks.

Motivation
----------
A SmolVLA action "chunk" is a short trajectory: T timesteps x A action dims.
We treat each chunk as a point cloud of T points in R^A and ask that the
*topological signature* of the predicted chunk match that of the expert chunk.

True persistent homology (0-dim, Vietoris-Rips) of a point cloud is governed
by the edge lengths of its minimum spanning tree -- i.e. by a particular sorted
subset of the pairwise distances. Computing exact PH each step (e.g. via gudhi)
is slow and its gradient is awkward. We instead use a cheap, fully
differentiable *surrogate*: the sorted top-k pairwise distances. Matching the
sorted distance spectra of predicted vs. expert chunks pushes the predicted
trajectory to have the same multi-scale "shape" (loops/spread/clustering) as the
expert, which is what the topological term is meant to encourage.

Properties
----------
* Differentiable: torch.cdist + torch.topk + L1/L2 are all differentiable w.r.t.
  the predicted actions (topk/sort just gather existing distance values).
* Batched: operates on (B, T, A) tensors with no Python loop over the batch.
* Cheap: O(B * T^2 * A) -- T is small (action horizon ~ 10-50).
* Training-time only: this is a regularizer added to the flow-matching loss;
  inference is completely unaffected.

Public API
----------
    ph_surrogate_loss(pred, target, k=8, p=2, reduction="mean") -> scalar/Tensor
    PHLoss(...)  # nn.Module wrapper for convenience

Run `python ph_loss.py` to execute the unit tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _sorted_topk_pairwise_distances(x: Tensor, k: int, p: float) -> Tensor:
    """
    For a batch of point clouds, return the k largest pairwise distances per
    cloud, sorted descending.

    Args
    ----
    x : (B, T, A) float tensor -- B point clouds of T points in R^A.
    k : number of largest pairwise distances to keep.
    p : Minkowski norm degree for the distance (2 = Euclidean).

    Returns
    -------
    (B, k) tensor of sorted (descending) top-k pairwise distances.
    Distances use only the upper triangle (i < j) so each pair is counted once.
    """
    if x.dim() != 3:
        raise ValueError(f"expected (B, T, A) input, got shape {tuple(x.shape)}")
    B, T, _ = x.shape
    if T < 2:
        raise ValueError(f"need at least 2 points per cloud, got T={T}")

    # torch.cdist has no bf16/fp16 CUDA kernel, and training runs in bf16.
    # Compute the (small) PH spectrum in fp32 -- numerically safer and the cast
    # is differentiable, so gradients flow back to the original-dtype tensor.
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()

    # Pairwise distance matrix: (B, T, T). cdist is differentiable.
    dmat = torch.cdist(x, x, p=p)  # (B, T, T)

    # Keep each unordered pair once via the strict upper triangle (i < j).
    iu, ju = torch.triu_indices(T, T, offset=1, device=x.device)
    pair_d = dmat[:, iu, ju]  # (B, P) where P = T*(T-1)/2

    P = pair_d.shape[1]
    kk = min(k, P)  # gracefully clamp if k exceeds available pairs
    topk_d, _ = torch.topk(pair_d, kk, dim=1, largest=True, sorted=True)  # (B, kk)
    return topk_d


def ph_surrogate_loss(
    pred: Tensor,
    target: Tensor,
    k: int = 8,
    p: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Differentiable PH surrogate loss between predicted and expert action chunks.

    The loss is the L1 distance between the sorted top-k pairwise-distance
    spectra of the predicted and target point clouds. It is zero exactly when
    the two clouds share the same top-k distance spectrum (same coarse topology
    / spread), and grows smoothly otherwise.

    Args
    ----
    pred   : (B, T, A) predicted action chunk (requires grad in training).
    target : (B, T, A) expert action chunk (no grad needed).
    k      : number of largest pairwise distances to match.
    p      : Minkowski degree for distances (2 = Euclidean).
    reduction : "mean" | "sum" | "none".

    Returns
    -------
    Scalar loss (reduction in {mean,sum}) or (B,) per-sample loss ("none").
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch pred {tuple(pred.shape)} vs target {tuple(target.shape)}")

    pred_spec = _sorted_topk_pairwise_distances(pred, k=k, p=p)            # (B, kk)
    with torch.no_grad():
        target_spec = _sorted_topk_pairwise_distances(target, k=k, p=p)   # (B, kk)

    # Per-sample L1 over the spectrum.
    per_sample = (pred_spec - target_spec).abs().mean(dim=1)  # (B,)

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError(f"unknown reduction '{reduction}'")


class PHLoss(nn.Module):
    """nn.Module wrapper so the loss can live in a policy and carry its config."""

    def __init__(self, k: int = 8, p: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.k = k
        self.p = p
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return ph_surrogate_loss(pred, target, k=self.k, p=self.p, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"k={self.k}, p={self.p}, reduction={self.reduction}"


# =============================================================================
# Unit tests  (run: python ph_loss.py)
# =============================================================================
def _run_tests() -> None:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ph_loss test] device = {dev}")

    B, T, A = 4, 16, 7  # typical: batch 4, horizon 16, 7-DoF action

    # 1. Identical chunks -> loss is exactly (numerically) zero.
    x = torch.randn(B, T, A, device=dev)
    loss_same = ph_surrogate_loss(x, x.clone())
    assert loss_same.item() < 1e-6, f"identical inputs should give ~0 loss, got {loss_same.item()}"
    print(f"[1] identical chunks loss = {loss_same.item():.2e}  (expect ~0)  OK")

    # 2. Different chunks -> strictly positive loss.
    y = torch.randn(B, T, A, device=dev)
    loss_diff = ph_surrogate_loss(x, y)
    assert loss_diff.item() > 0, "different inputs should give positive loss"
    print(f"[2] different chunks loss = {loss_diff.item():.4f}  (expect >0)  OK")

    # 3. Differentiable: gradient flows to `pred`, not to `target`.
    pred = torch.randn(B, T, A, device=dev, requires_grad=True)
    target = torch.randn(B, T, A, device=dev)
    loss = ph_surrogate_loss(pred, target)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "grad must exist & be finite"
    assert pred.grad.abs().sum().item() > 0, "grad should be non-zero"
    print(f"[3] grad wrt pred: finite={torch.isfinite(pred.grad).all().item()}, "
          f"||grad||={pred.grad.norm().item():.4f}  OK")

    # 4. Gradient descent on the surrogate actually reduces it (sanity of signal).
    pred2 = torch.randn(B, T, A, device=dev, requires_grad=True)
    target2 = torch.randn(B, T, A, device=dev)
    opt = torch.optim.SGD([pred2], lr=0.05)
    l0 = ph_surrogate_loss(pred2, target2).item()
    for _ in range(200):
        opt.zero_grad()
        l = ph_surrogate_loss(pred2, target2)
        l.backward()
        opt.step()
    l1 = ph_surrogate_loss(pred2, target2).item()
    assert l1 < l0, f"optimization should reduce loss: {l0:.4f} -> {l1:.4f}"
    print(f"[4] optimize surrogate: {l0:.4f} -> {l1:.4f}  (decreasing)  OK")

    # 5. Reductions & k-clamping behave.
    none_out = ph_surrogate_loss(x, y, reduction="none")
    assert none_out.shape == (B,), none_out.shape
    big_k = ph_surrogate_loss(x, y, k=10_000)  # k >> available pairs -> clamped
    assert torch.isfinite(big_k).all()
    print(f"[5] reduction='none' shape={tuple(none_out.shape)}, k-clamp ok ({big_k.item():.4f})  OK")

    # 6. bfloat16 path (matches training dtype) stays finite.
    xb = x.to(torch.bfloat16)
    yb = y.to(torch.bfloat16)
    lb = ph_surrogate_loss(xb, yb)
    assert torch.isfinite(lb).all(), "bf16 loss must be finite"
    print(f"[6] bfloat16 loss = {lb.item():.4f}  (finite)  OK")

    print("\n[ph_loss test] ALL TESTS PASSED")


if __name__ == "__main__":
    _run_tests()
