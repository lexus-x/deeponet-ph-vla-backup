"""
run_experiments.py
==================
Quantitative mechanism results for the three contributions. Outputs tables to
stdout + results/*.png + results/results_raw.json.

Exp A  Chunk-execution strategies (numpy simulation of the closed-loop):
       naive switching vs fractional-tau ensembling vs boundary pinning.
       Metrics: boundary jump, tracking RMSE, jerk. (The torch implementation
       of the strategies is validated separately in operator_chunk_exec.py.)

Exp B  Topological-loss stability: Ayush's top-k surrogate vs TRUE H0 vs
       H0 + PCA-3, on chunks embedded in padded R^32 (the SmolVLA setting).
       Metrics: separability AUC (perturbed-same vs different) and
       warp-tolerance ratio (a good shape regularizer should not punish
       small time-warps of the same motion).

Exp C  POD eigen-motion analysis of action chunks: variance explained vs #modes,
       reconstruction error of POD vs Fourier vs random basis at equal p.

Uses real LIBERO-Spatial chunks (data/chunks_spatial.npy) if present,
else synthetic smooth chunks (clearly labelled).
"""

from __future__ import annotations

import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "git-repo", "DeepONet PH", "v2"))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
OUT = {}

# ---------------------------------------------------------------- data
def load_chunks(T=50):
    path = os.path.join(HERE, "data", "chunks_spatial.npy")
    if os.path.exists(path):
        c = np.load(path)
        if c.shape[1] != T:
            c = c[:, :T]
        return c.astype(np.float64), "LIBERO-Spatial (real demos)"
    # synthetic fallback: smooth low-rank trajectories, 7 dof
    N, K, A = 2000, 8, 7
    t = np.linspace(0, 1, T)
    modes = np.stack([np.sin((k + 1) * np.pi * t + rng.uniform(0, 2 * np.pi))
                      for k in range(K)])
    W = rng.normal(size=(N, K)) * (1.0 / (1 + np.arange(K)))
    M = rng.normal(size=(K, A)) * 0.5
    chunks = np.einsum("nk,kt,ka->nta", W, modes, M) + 0.01 * rng.normal(size=(N, T, A))
    return chunks, "synthetic smooth chunks (fallback)"


# =============================================================== Exp A
def exp_a(n_seeds=20, E=400, H=50, R=25, A=7, ctrl_per_step=2):
    """Simulate closed-loop chunk execution with systematic per-replan error."""
    def smooth_curve(T, amp, rng, n_basis=6):
        t = np.linspace(0, 1, T)
        c = rng.normal(size=(n_basis, A)) * amp / np.sqrt(n_basis)
        return sum(c[k] * np.sin((k + 1) * np.pi * t)[:, None] for k in range(n_basis))

    strategies = ["naive", "ensemble", "pinned", "ensemble+pinned"]
    met = {s: {"jump": [], "rmse": [], "jerk": []} for s in strategies}

    for seed in range(n_seeds):
        r = np.random.default_rng(seed)
        tgrid = np.arange(E) / H
        gt = np.stack([np.sum([a * np.sin(2 * np.pi * f * tgrid + ph)
                               for a, f, ph in zip(r.uniform(0.2, 1, 4),
                                                    r.uniform(0.2, 1.5, 4),
                                                    r.uniform(0, 2 * np.pi, 4))], axis=0)
                       for _ in range(A)]).T                       # (E, A)

        replans = list(range(0, E - H, R))
        # each replan: predicted chunk = gt segment + systematic smooth error
        chunk_err = {s: smooth_curve(H, 0.15, r) for s in replans}
        def chunk_val(s, tau):                                     # tau in [0,1)
            pos = s + tau * (H - 1)
            i0 = int(np.clip(np.floor(pos), 0, E - 2)); w = pos - i0
            g = gt[i0] * (1 - w) + gt[i0 + 1] * w
            j0 = int(np.clip(np.floor(tau * (H - 1)), 0, H - 2)); v = tau * (H - 1) - j0
            e = chunk_err[s][j0] * (1 - v) + chunk_err[s][j0 + 1] * v
            return g + e

        n_ctrl = (E - H) * ctrl_per_step
        for strat in strategies:
            traj, jumps = [], []
            prev_a, cur_prev = None, None
            pin_offset = np.zeros(A)
            pin_total = max(1, int(0.25 * H * ctrl_per_step))
            pin_left = 0
            for ic in range(n_ctrl):
                t_step = ic / ctrl_per_step
                active = [s for s in replans if s <= t_step < s + H]
                cur = max(active)
                is_boundary = cur_prev is not None and cur != cur_prev
                cur_prev = cur

                if "ensemble" in strat:
                    ws, preds = [], []
                    for s in active[-4:]:
                        tau = (t_step - s) / H
                        preds.append(chunk_val(s, tau))
                        ws.append(np.exp(-0.15 * (t_step - s)))
                    ws = np.array(ws) / np.sum(ws)
                    a = np.einsum("k,ka->a", ws, np.stack(preds))
                else:
                    a = chunk_val(cur, (t_step - cur) / H)

                if "pinned" in strat:
                    if prev_a is not None and is_boundary:
                        pin_offset = prev_a - a   # continuity at the switch
                        pin_left = pin_total
                    if pin_left > 0:
                        ramp = 0.5 * (1 + np.cos(np.pi * (1 - pin_left / pin_total)))
                        a = a + pin_offset * ramp
                        pin_left -= 1

                if prev_a is not None and is_boundary:
                    jumps.append(np.linalg.norm(a - prev_a))
                traj.append(a); prev_a = a
            traj = np.stack(traj)
            gt_ctrl = np.stack([gt[min(int(ic / ctrl_per_step), E - 1)] for ic in range(n_ctrl)])
            met[strat]["jump"].append(np.mean(jumps) if jumps else 0.0)
            met[strat]["rmse"].append(float(np.sqrt(np.mean((traj - gt_ctrl) ** 2))))
            met[strat]["jerk"].append(float(np.mean(np.abs(np.diff(traj, 2, axis=0)))))

    print("\n== Exp A: chunk-execution strategies "
          f"(sim, {n_seeds} seeds, H={H}, replan={R}) ==")
    print(f"{'strategy':<18}{'boundary jump':>15}{'tracking RMSE':>15}{'jerk':>12}")
    OUT["exp_a"] = {}
    for s in strategies:
        j, rm, jk = (np.array(met[s][k]) for k in ("jump", "rmse", "jerk"))
        print(f"{s:<18}{j.mean():>11.4f}±{j.std():.3f}"
              f"{rm.mean():>11.4f}±{rm.std():.3f}{jk.mean():>9.5f}±{jk.std():.4f}")
        OUT["exp_a"][s] = {"jump": [j.mean(), j.std()],
                           "rmse": [rm.mean(), rm.std()],
                           "jerk": [jk.mean(), jk.std()]}


# =============================================================== Exp B
def exp_b(chunks, label, n_pairs=300, pad_to=32):
    from ph_loss import ph_surrogate_loss                  # Ayush's surrogate
    from ph_loss_stable import ph_h0_loss

    N, T, A = chunks.shape
    sd = chunks.std(axis=(0, 1))                            # per-dim std

    def pad(x):                                             # (n,T,A)->(n,T,32)
        out = np.zeros((x.shape[0], T, pad_to)); out[..., :A] = x
        return out

    idx = rng.choice(N, size=n_pairs, replace=False)
    base = chunks[idx]
    # "same": small iid noise on real dims + pred-like noise on pad dims
    pert = base + rng.normal(size=base.shape) * (0.05 * sd)
    # "warped": same motion, mild time-warp + 5% amplitude scale (should be cheap)
    tw = np.linspace(0, 1, T)
    warp_t = np.clip(tw + 0.05 * np.sin(np.pi * tw), 0, 1)
    warped = np.stack([np.stack([np.interp(warp_t, tw, b[:, d]) for d in range(A)], 1)
                       for b in base]) * 1.05
    # "diff": a different chunk
    diff = chunks[rng.permutation(idx)]

    base_p = torch.tensor(pad(base), dtype=torch.float32)
    pad_noise = lambda x: torch.tensor(
        pad(x), dtype=torch.float32) + torch.cat(
        [torch.zeros(n_pairs, T, A),
         0.02 * torch.randn(n_pairs, T, pad_to - A)], dim=-1)
    pert_p, warp_p, diff_p = pad_noise(pert), pad_noise(warped), pad_noise(diff)

    losses = {
        "surrogate_R32 (Ayush v2)":
            lambda a, b: ph_surrogate_loss(a, b, k=8, reduction="none"),
        "true H0, R32":
            lambda a, b: ph_h0_loss(a, b, k=8, proj_dim=None, reduction="none"),
        "true H0 + PCA-3 (ours)":
            lambda a, b: ph_h0_loss(a, b, k=8, proj_dim=3, reduction="none"),
        "surrogate + PCA-3 (ablation)":
            lambda a, b: ph_surrogate_loss(
                *(__import__("ph_loss_stable")._pca_project(a, b, 3)),
                k=8, reduction="none"),
    }

    print(f"\n== Exp B: topological-loss stability ({label}, "
          f"{n_pairs} pairs, padded R^{pad_to}) ==")
    print(f"{'loss':<30}{'AUC same<diff':>14}{'warp/diff ratio':>17}")
    OUT["exp_b"] = {}
    for name, fn in losses.items():
        with torch.no_grad():
            l_same = fn(pert_p, base_p).numpy()
            l_warp = fn(warp_p, base_p).numpy()
            l_diff = fn(diff_p, base_p).numpy()
        # AUC: P(loss_same < loss_diff) over random pairs
        auc = float(np.mean(l_same[:, None] < l_diff[None, :]))
        ratio = float(np.median(l_warp) / (np.median(l_diff) + 1e-12))
        print(f"{name:<30}{auc:>14.3f}{ratio:>17.3f}")
        OUT["exp_b"][name] = {"auc": auc, "warp_diff_ratio": ratio,
                              "median_same": float(np.median(l_same)),
                              "median_warp": float(np.median(l_warp)),
                              "median_diff": float(np.median(l_diff))}
    print("(AUC=1 ideal: perturbed-same always cheaper than different chunk;"
          " lower warp/diff = more warp-tolerant)")


# =============================================================== Exp C
def exp_c(chunks, label):
    N, T, A = chunks.shape
    ntr = int(0.8 * N)
    perm = rng.permutation(N)
    tr, te = chunks[perm[:ntr]], chunks[perm[ntr:]]

    Xtr = tr.reshape(ntr, -1); Xte = te.reshape(len(te), -1)
    mu = Xtr.mean(0)
    U, S, Vh = np.linalg.svd(Xtr - mu, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    cum = np.cumsum(var)
    marks = {q: int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99, 0.999)}
    print(f"\n== Exp C: POD eigen-motions ({label}, {N} chunks {T}x{A}) ==")
    print("modes for variance:", {f"{k:.1%}": v for k, v in marks.items()})

    def recon_err(basis, p):
        B = basis[:p]
        Z = (Xte - mu) @ B.T
        R = Z @ B + mu
        return float(np.sqrt(np.mean((R - Xte) ** 2)))

    # competitor bases at equal p: DCT (per-dim temporal) and random orthonormal
    import numpy.linalg as la
    D = Xtr.shape[1]
    dct = np.zeros((D, D))
    # separable temporal-DCT basis over (T,A) flattened: mode (k, dim d)
    k_t = np.arange(T)
    rows = []
    for k in range(T):
        for d in range(A):
            v = np.zeros((T, A))
            v[:, d] = np.cos(np.pi * k * (2 * k_t + 1) / (2 * T))
            rows.append(v.reshape(-1))
    dct = np.stack(rows)
    dct /= la.norm(dct, axis=1, keepdims=True)
    rnd = la.qr(rng.normal(size=(D, D)))[0].T

    ps = [4, 8, 16, 32, 64, 128]
    print(f"{'p':>5}{'POD rmse':>12}{'DCT rmse':>12}{'random rmse':>13}")
    OUT["exp_c"] = {"modes_for_variance": {f"{k:.3f}": v for k, v in marks.items()},
                    "recon": {}}
    for p in ps:
        e_pod, e_dct, e_rnd = recon_err(Vh, p), recon_err(dct, p), recon_err(rnd, p)
        print(f"{p:>5}{e_pod:>12.5f}{e_dct:>12.5f}{e_rnd:>13.5f}")
        OUT["exp_c"]["recon"][p] = {"pod": e_pod, "dct": e_dct, "random": e_rnd}

    # plots: spectrum + top-6 eigen-motions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(var[:100]); axes[0].set_title("POD spectrum (variance share)")
    axes[0].set_xlabel("mode"); axes[0].grid(alpha=0.3)
    t = np.linspace(0, 1, T)
    for k in range(6):
        m = Vh[k].reshape(T, A)
        axes[1].plot(t, m[:, 0], label=f"mode {k}")
    axes[1].set_title("top-6 eigen-motions (action dim 0)")
    axes[1].set_xlabel("tau"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "expC_eigenmotions.png"), dpi=130)
    print("plot -> results/expC_eigenmotions.png")


if __name__ == "__main__":
    chunks, label = load_chunks()
    print(f"data: {label}, chunks={chunks.shape}")
    exp_a()
    exp_b(chunks, label)
    exp_c(chunks, label)
    with open(os.path.join(RES, "results_raw.json"), "w") as f:
        json.dump(OUT, f, indent=2)
    print("\nraw numbers -> results/results_raw.json")
