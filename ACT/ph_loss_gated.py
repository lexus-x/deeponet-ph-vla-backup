"""
ph_loss_gated.py
================
GATED persistent-homology loss: "use it only when needed, else idle".

Two gates (both must open for PH to be applied to a sample):
  1. WARMUP gate (global): PH stays fully idle for the first `warmup_steps`
     optimiser steps — the regression head learns the rough trajectory first.
  2. TRIGGER gate (per-sample): after warmup, compute the cheap topological
     spectrum of pred vs target; apply the PH penalty ONLY to samples whose
     normalised topological divergence exceeds `trigger`. Samples that are
     already topologically close contribute nothing (the loss for them is 0),
     so PH is "idle" exactly when the shape is already right.

If, on a given step, no sample triggers, the whole PH term is a hard 0 and the
expensive gradient path is skipped — so it costs ~nothing when not needed.

Returns (loss, info) where info has the live gate stats for logging.
"""
from __future__ import annotations
import torch
from torch import Tensor

from ph_loss import _sorted_topk_pairwise_distances


@torch.no_grad()
def _divergence(pred: Tensor, target: Tensor, k: int, p: float) -> Tensor:
    """Per-sample normalised topological divergence in [0, inf) (no grad)."""
    ps = _sorted_topk_pairwise_distances(pred, k=k, p=p)      # (B, kk)
    ts = _sorted_topk_pairwise_distances(target, k=k, p=p)    # (B, kk)
    num = (ps - ts).abs().mean(dim=1)
    den = ts.abs().mean(dim=1).clamp(min=1e-6)
    return num / den                                          # (B,)


def gated_ph_loss(pred: Tensor, target: Tensor, *, step: int, warmup_steps: int = 5000,
                  trigger: float = 0.15, k: int = 8, p: float = 2.0):
    """Trigger-gated PH surrogate. pred/target: (B, T, A). Returns (loss, info)."""
    B = pred.shape[0]
    dev = pred.device
    if step < warmup_steps:
        return pred.new_zeros(()), {"ph_active_frac": 0.0, "ph_gate": "warmup", "ph_raw": 0.0}

    div = _divergence(pred, target, k, p)                     # (B,)
    mask = (div > trigger).float()                            # (B,) which samples need PH
    active = float(mask.mean().item())
    if mask.sum() == 0:
        return pred.new_zeros(()), {"ph_active_frac": 0.0, "ph_gate": "idle", "ph_raw": 0.0}

    # differentiable surrogate, applied per-sample, masked to triggered samples only
    ps = _sorted_topk_pairwise_distances(pred, k=k, p=p)      # (B, kk) WITH grad
    with torch.no_grad():
        ts = _sorted_topk_pairwise_distances(target, k=k, p=p)
    per_sample = (ps - ts).pow(2).mean(dim=1)                 # (B,)
    loss = (per_sample * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, {"ph_active_frac": active, "ph_gate": "active", "ph_raw": float(loss.item())}
