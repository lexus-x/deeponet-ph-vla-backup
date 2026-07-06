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

### Provenance — the replan bug (before → after)

**The error.** The first pi0.5 run used the model config's default `n_action_steps=50`: it executed all 50
predicted actions **open‑loop** before re‑querying the policy, instead of the **replan=5** receding‑horizon
control used by the SmolVLA study this campaign compares against. Near‑open‑loop control tanks closed‑loop
LIBERO tasks, so every absolute score was artificially low and the setup was not comparable to SmolVLA
(though internally fair — both heads ran at 50). A second audit also caught an asymmetric `--init` (the flow
baseline had been discarding its pretrain).

**The change.** `eval_pi05.py` now forces `policy.config.n_action_steps = args.replan` (=5) so the policy
re‑plans every 5 steps, matching SmolVLA; the `--init` asymmetry was fixed so flow and DeepONet resume the
same pretrain. The **entire campaign was re‑run** at replan=5 → `results_replan5/` (canonical).

**Before — open‑loop (`n_action_steps=50`), superseded:**

| Suite | Flow in‑dist / **Plus** | DeepONet in‑dist / **Plus** | +PH in‑dist / **Plus** |
|---|---|---|---|
| Spatial | 53.3 / 33.9 | 63.3 / 19.6 | 70.0 / 16.1 |
| Object | 58.3 / 28.6 | 41.7 / 7.1 | 47.5 / 8.9 |
| Long | 28.3 / 14.3 | 30.8 / 5.4 | 30.8 / 7.1 |
| Goal | 71.7 / 33.9 | 77.5 / 32.1 | 78.3 / 26.8 |
| **Average** | **52.9 / 27.7** | **53.3 / 16.1** | **56.7 / 14.7** |

**After — replan=5 (canonical):** the results table at the top of this README
(Flow **84.0 / 54.9**, DeepONet **81.5 / 29.5**, +PH **76.3 / 35.7**).

**What the fix changed — and didn't.** The fix roughly *doubled* every absolute score (Flow in‑dist 52.9 → 84.0),
confirming the bug was crippling. But the **conclusion is unchanged**: flow beats DeepONet on robustness under
**both** protocols (Plus average — Flow 27.7 vs DeepONet 16.1 open‑loop; 54.9 vs 29.5 at replan=5). The negative
result is a property of the frozen head‑only regime, **not** an artifact of the replan bug.

## Checkpoints
The preserved deeponet_ph 40‑task pretrain (16 GB) is backed up to HF
`AyushShah1107/pi05-deeponet-libero` under `checkpoints/`. Per‑suite finetune checkpoints are pruned after
each eval (data‑loss‑safe: only after a confirmed result JSON).
