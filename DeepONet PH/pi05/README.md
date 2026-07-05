# Comp‑1 — DeepONet operator head on pi0.5 (LIBERO)

Scaling the DeepONet operator action head (+ PH loss) to **pi0.5 (3.3 B, PaliGemma VLM + flow‑matching
action expert)**. Head‑only (frozen backbone): 40‑task combined **pretrain (15K)** → per‑suite **finetune
(15K)** → **in‑dist + LIBERO‑Plus** on Spatial/Object/Long/Goal. Three variants: `c1_pi05` (flow baseline) ·
`c1_pi05_deeponet` · `c1_pi05_deeponet_ph`.

Code: `train_pi05.py`, `eval_pi05.py`, `modeling_pi05_deeponet.py`. Runner: `../comp3_groot/run_c1_replan5.sh`.

## Results — closed‑loop **replan=5**, single seed (`results_replan5/`)

| Suite | Flow in‑dist / **Plus** | DeepONet in‑dist / **Plus** | +PH in‑dist / **Plus** |
|---|---|---|---|
| Spatial | 88.3 / **64.3** | 86.7 / 39.3 | 90.0 / 48.2 |
| Object | 87.5 / **55.4** | 81.7 / 30.4 | 83.3 / 32.1 |
| Long | 66.7 / **50.0** | 60.8 / 12.5 | 44.2 / 12.5 |
| Goal | 93.3 / **50.0** | 96.7 / 35.7 | 87.5 / 50.0 |
| **Average** | **84.0 / 54.9** | 81.5 / 29.5 | 76.3 / 35.7 |

**Honest result: on pi0.5, flow leads both in‑dist (+2.5 pp) and robustness (+25.4 pp).** The SmolVLA/ACT
robustness win did **not** transfer. This is a real negative result, not an artifact — the eval was audited
(closed‑loop replan=5, correct action space + physics‑settle, symmetric flow‑vs‑DeepONet `--init`, matched
aggregation, single seed).

### Why — frozen backbone
On pi0.5 the backbone is **frozen and head‑only**, and was pretrained end‑to‑end *as a flow model*, so its
features are native to flow and foreign to a bolted‑on operator head. The operator head's benefit is
representation‑level, and it can't reshape a frozen representation. The signature fits: **in‑dist nearly ties**
(the readout still fits) while **Plus collapses** (OOD robustness lives in the representation). In the cases where
the operator head *won* robustness (ACT trains end‑to‑end; SmolVLA unfreezes stage‑2 at lr 1e‑5), the
representation could co‑adapt. **The decisive follow‑up is unfreezing the pi0.5 backbone** (matching the SmolVLA
regime) on 1–2 suites.

### Provenance note — the replan bug
An earlier open‑loop run used pi0.5's config default `n_action_steps=50` (nearly open‑loop) instead of the
intended **replan=5** used by the SmolVLA study. Both heads ran at 50, so it was internally fair but not
comparable to SmolVLA. `eval_pi05.py` now sets `policy.config.n_action_steps = args.replan`; **`results_replan5/`
is the canonical, comparable set** (the open‑loop numbers are superseded). A second audit also fixed an asymmetric
`--init` (the flow baseline had been discarding its pretrain) — fixed before this run, so the comparison is fair.

## Checkpoints
The preserved deeponet_ph 40‑task pretrain (16 GB) is backed up to HF
`AyushShah1107/pi05-deeponet-libero` under `checkpoints/`. Per‑suite finetune checkpoints are pruned after
each eval (data‑loss‑safe: only after a confirmed result JSON).
