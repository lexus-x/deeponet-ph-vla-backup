"""
operator_bid.py
================
DeepONet-native Bidirectional Decoding (Operator-BID).

Unlike diffusion BID (multi-noise samples), DeepONet is deterministic. We obtain
K candidate action chunks by injecting small Gaussian noise into proprioceptive
state, then select the candidate whose a[0] is closest to the last executed
action (backward coherence). This is a *new method* — not averaging (ens) and
not hard pinning — that exploits multi-hypothesis search at test time.

Reference framing: BID (Liu et al., ICLR 2025) for chunked policies; we adapt
the selection criterion to SmolVLA/DeepONet without requiring a likelihood model.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


STATE_KEYS = (
    "observation.state",
    "observation.agent_pos",
    "state",
    "agent_pos",
)


def _clone_batch(pin: dict) -> dict:
    out = {}
    for k, v in pin.items():
        out[k] = v.clone() if torch.is_tensor(v) else v
    return out


def _perturb_state(pin: dict, noise_std: float, generator: torch.Generator | None = None) -> dict:
    """Add isotropic Gaussian noise to the first matching proprio key."""
    out = _clone_batch(pin)
    for key in STATE_KEYS:
        if key in out and torch.is_tensor(out[key]):
            x = out[key]
            eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            out[key] = x + noise_std * eps
            return out
    # fallback: perturb any 1D/2D float tensor that looks like state (dim 7..32)
    for key, v in out.items():
        if torch.is_tensor(v) and v.dtype.is_floating_point and v.shape[-1] in range(7, 33):
            eps = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)
            out[key] = v + noise_std * eps
            return out
    return out


@torch.no_grad()
def select_bid_chunk(
    predict_fn,
    pin: dict,
    last_action: Tensor | None,
    k: int = 8,
    noise_std: float = 0.01,
) -> Tensor:
    """Return (T, A) CPU float chunk selected by backward coherence.

    predict_fn(pin_dict) -> (T, A) CPU tensor
    """
    clean = predict_fn(pin)
    if last_action is None or k <= 1:
        return clean

    a_prev = last_action.reshape(-1).float()
    best = clean
    best_dist = torch.norm(clean[0].float() - a_prev).item()

    for i in range(k - 1):
        noisy = _perturb_state(pin, noise_std)
        cand = predict_fn(noisy)
        dist = torch.norm(cand[0].float() - a_prev).item()
        if dist < best_dist:
            best_dist = dist
            best = cand
    return best
