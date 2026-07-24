"""
operator_chunk_exec.py
======================
Analytic chunk-execution toolkit for a DeepONet action head.

Because the operator head represents a chunk as a continuous function
a(tau) = OutMLP(c * Trunk(tau)), three chunk-execution problems that need
special machinery for flow/diffusion heads (RTC inpainting, BID resampling,
finite-difference phase detection) become closed-form here:

  1. query_tau     — evaluate a cached chunk code c at ANY real tau
                     (no backbone pass; Trunk+OutMLP only).
  2. ChunkEnsembler— ACT-style temporal ensembling with EXACT fractional-tau
                     alignment across overlapping chunks (discrete chunks can
                     only align to the nearest grid step).
  3. pin_boundary  — hard chunk-boundary continuity a_new(0) = a_old(t_switch)
                     via an offset that decays over the new chunk.
  4. speed_profile / replan_points — |da/dtau| by autodiff through the trunk;
                     low-speed replan points are minima of an analytic profile
                     (PACE finds them by finite differences on discrete actions).

Drop-in: wraps a DeepONetHeadV2 instance (deeponet_head_v2.py) untouched.
Run `python operator_chunk_exec.py` for the self-check.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class OperatorChunkPolicy(nn.Module):
    """Wraps DeepONetHeadV2 to expose the continuous-time interface.

    The base head computes  a(tau) = OutMLP(c * trunk(fourier(tau))) + bias
    on a fixed tau grid. Here we split that into encode() (one backbone-side
    pass -> cached code c) and query() (cheap trunk-side evaluation at any tau).
    """

    def __init__(self, head):
        super().__init__()
        self.head = head

    @torch.no_grad()
    def encode(self, prefix: Tensor, pad_mask: Tensor) -> Tensor:
        """One expensive pass: prefix tokens -> branch code c (B, p)."""
        ctx = self.head.pool(prefix, pad_mask)
        return self.head.branch(ctx.flatten(1))

    def query(self, c: Tensor, tau: Tensor) -> Tensor:
        """Cheap pass: code (B,p) + tau (Q,) in [0,1] -> actions (B,Q,A).

        Differentiable in tau, so autodiff gives da/dtau exactly.
        """
        tau = tau.reshape(-1, 1).to(c.dtype)
        phi = self.head.trunk(self.head._fourier(tau))        # (Q, p)
        feat = c.unsqueeze(1) * phi.unsqueeze(0)               # (B, Q, p)
        return self.head.out_mlp(feat) + self.head.out_bias    # (B, Q, A)

    def speed_profile(self, c: Tensor, tau: Tensor) -> Tensor:
        """|da/dtau| (B, Q) via autodiff — the analytic speed of the chunk."""
        tau = tau.detach().clone().requires_grad_(True)
        a = self.query(c, tau)                                  # (B, Q, A)
        # sum over batch+action dims: d(sum a)/d tau_q = per-query velocity sums
        grads = torch.autograd.grad(a.sum(), tau, create_graph=False)[0]  # (Q,)
        # per-sample exact velocity needs a vmap/jacobian; for replan-point
        # selection B==1 in deployment, so compute per-dim jacobian cheaply:
        if c.shape[0] == 1:
            tau2 = tau.detach().clone().requires_grad_(True)
            a2 = self.query(c, tau2)[0]                         # (Q, A)
            v = []
            for d in range(a2.shape[-1]):
                g = torch.autograd.grad(a2[:, d].sum(), tau2, retain_graph=True)[0]
                v.append(g)
            vel = torch.stack(v, dim=-1)                        # (Q, A)
            return vel.norm(dim=-1, keepdim=False).unsqueeze(0)  # (1, Q)
        return grads.abs().unsqueeze(0).expand(c.shape[0], -1)

    def replan_point(self, c: Tensor, tau_min: float = 0.3, tau_max: float = 0.9,
                     n_grid: int = 64) -> float:
        """Best (lowest-speed) replan point in [tau_min, tau_max] — PACE-style
        phase-aware boundary, but on an analytic speed profile."""
        taus = torch.linspace(tau_min, tau_max, n_grid)
        speed = self.speed_profile(c, taus)[0]                  # (Q,)
        return float(taus[int(speed.argmin())])


class ChunkEnsembler:
    """ACT-style temporal ensembling with exact fractional-tau alignment.

    Keeps the last `max_chunks` (code, start_time) pairs. At wall-clock step t,
    every retained chunk k is queried at its OWN tau_k = (t - start_k)/H, and
    predictions are blended with exponential weights w ~ exp(-alpha * age).
    Cost: one trunk+out_mlp evaluation per retained chunk (no backbone pass).
    """

    def __init__(self, policy: OperatorChunkPolicy, horizon_steps: int,
                 alpha: float = 0.5, max_chunks: int = 4):
        self.policy = policy
        self.H = horizon_steps
        self.alpha = alpha
        self.max_chunks = max_chunks
        self.chunks: list[tuple[Tensor, int]] = []  # (code c (1,p), start_step)

    def add_chunk(self, c: Tensor, start_step: int) -> None:
        self.chunks.append((c, start_step))
        self.chunks = self.chunks[-self.max_chunks:]

    def act(self, step: int) -> Tensor:
        """Blended action at integer (or fractional) wall-clock step."""
        preds, ws = [], []
        for c, s in self.chunks:
            tau = (step - s) / self.H
            if 0.0 <= tau <= 1.0:
                a = self.policy.query(c, torch.tensor([tau]))[:, 0]  # (1, A)
                preds.append(a)
                ws.append(torch.exp(torch.tensor(-self.alpha * (step - s))))
        if not preds:
            raise RuntimeError("no active chunk covers this step")
        w = torch.stack(ws); w = w / w.sum()
        return (torch.stack(preds, 0) * w.view(-1, 1, 1)).sum(0)  # (1, A)


def pin_boundary(a_new_fn, a_old_at_switch: Tensor, decay_tau: float = 0.25):
    """Hard boundary continuity: returns a'(tau) = a_new(tau) + offset(tau) with
    a'(0) == a_old(t_switch) exactly, offset decaying to 0 by tau=decay_tau.

    a_new_fn: callable tau (Q,) -> (B, Q, A). Zero training cost, exact at tau=0.
    """
    a_new_0 = a_new_fn(torch.tensor([0.0]))[:, 0]              # (B, A)
    offset0 = a_old_at_switch - a_new_0                        # (B, A)

    def pinned(tau: Tensor) -> Tensor:
        a = a_new_fn(tau)                                       # (B, Q, A)
        # smooth cosine ramp 1 -> 0 on [0, decay_tau]
        w = torch.clamp(tau / decay_tau, 0.0, 1.0)
        ramp = 0.5 * (1.0 + torch.cos(torch.pi * w))            # (Q,)
        return a + offset0.unsqueeze(1) * ramp.view(1, -1, 1)

    return pinned


# ---------------------------------------------------------------- self-check
def _self_check():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "git-repo", "DeepONet PH", "v2"))
    from deeponet_head_v2 import DeepONetHeadV2

    torch.manual_seed(0)
    B, N, D, T, A = 1, 64, 960, 50, 32
    head = DeepONetHeadV2(context_dim=D, chunk_size=T, action_dim=A)
    pol = OperatorChunkPolicy(head)
    prefix = torch.randn(B, N, D); mask = torch.ones(B, N, dtype=torch.bool)

    # 1. query at the training grid == head.forward
    c = pol.encode(prefix, mask)
    grid = torch.linspace(0, 1, T)
    a_q = pol.query(c, grid)
    a_f = head(prefix, mask)
    assert torch.allclose(a_q, a_f, atol=1e-5), "query(grid) must equal forward()"
    print("[1] query(grid) == forward()                       OK")

    # 2. fractional tau works and is continuous
    a_mid = pol.query(c, torch.tensor([0.5, 0.5 + 1e-4]))
    assert (a_mid[:, 0] - a_mid[:, 1]).abs().max() < 1e-2, "a(tau) not continuous"
    print("[2] fractional-tau query continuous                OK")

    # 3. analytic speed profile finite, replan point in range
    tau_star = pol.replan_point(c)
    assert 0.3 <= tau_star <= 0.9
    print(f"[3] analytic replan point tau*={tau_star:.3f}         OK")

    # 4. boundary pinning: exact continuity at tau=0
    a_old = pol.query(c, torch.tensor([0.7]))[:, 0] + 0.3      # pretend old chunk
    pinned = pin_boundary(lambda t: pol.query(c, t), a_old)
    a0 = pinned(torch.tensor([0.0]))[:, 0]
    assert torch.allclose(a0, a_old, atol=1e-5), "boundary not pinned"
    a_late = pinned(torch.tensor([0.5]))[:, 0]
    a_raw = pol.query(c, torch.tensor([0.5]))[:, 0]
    assert torch.allclose(a_late, a_raw, atol=1e-5), "offset must decay to 0"
    print("[4] boundary pinned exactly, offset decays          OK")

    # 5. ensembler blends overlapping chunks
    ens = ChunkEnsembler(pol, horizon_steps=T, alpha=0.1)
    ens.add_chunk(c, start_step=0)
    ens.add_chunk(pol.encode(torch.randn(B, N, D), mask), start_step=25)
    a_blend = ens.act(step=30)
    assert a_blend.shape == (1, A) and torch.isfinite(a_blend).all()
    print("[5] fractional-tau ensembling                      OK")
    print("\n[operator_chunk_exec] ALL CHECKS PASSED")


if __name__ == "__main__":
    _self_check()
