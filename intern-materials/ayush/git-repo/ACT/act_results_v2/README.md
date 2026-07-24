# ACT V2 — transfer campaign (40-task pretrain → per-suite finetune)

The **V2 regime** for the ACT DeepONet head: a 40-task multi-suite **pretrain (15K)** followed by a
per-suite **finetune** (8K, then re-finetuned to 15K), vs V1's from-scratch 30K per suite. Same eval
across V1/V2 (`evaluate_act.py`, 10 tasks × 3 seeds, replan=5, max_steps 520, last checkpoint, EMA 0.999).

Variants: `act` (baseline) · `act_deeponet` · `act_deeponet_ph`.

## Results — V2, 15K finetune (`summary_v2_15k.csv`, parsed from `eval_v2_15k.log`)

| Suite | act in‑dist / **Plus** | act_deeponet in‑dist / **Plus** | +PH in‑dist / **Plus** | DeepONet − baseline (in‑dist) |
|---|---|---|---|---|
| Spatial | 81.0 / 58.3 | **85.3 / 66.7** | 76.7 / 57.1 | **+4.3** ✅ |
| Object  | 81.3 / 45.2 | 65.0 / 45.2 | 70.7 / 38.1 | **−16.3** ❌ |
| Long    | 45.3 / 19.0 | **66.3 / 20.2** | 44.3 / 23.8 | **+21.0** ✅ |
| Goal    | 86.0 / 50.0 | 84.7 / 53.6 | 74.0 / 50.0 | −1.3 (tie) |
| **Avg** | 73.4 / 43.1 | **75.3 / 46.4** | 66.4 / 44.8 | **+1.9 / +3.3** |

**Honest read:** under V2, the DeepONet head **beats the ACT baseline on the whole-suite average**
(in‑dist +1.9, robustness +3.3) and wins Spatial and Long convincingly — but **Object flips negative
(−16.3 in‑dist)**, the opposite of V1 (where DeepONet won Object +5.3).

### The Object sign‑flip — root cause: training‑regime under‑training, not a bug
Eval is byte‑identical between V1 and V2, so it is not an eval/harness artifact. The only difference is
budget: **V1 = 30K from‑scratch on Object; V2 = 15K pretrain → 8K/15K finetune.** The operator head
**under‑trains on Object** under the short transfer‑finetune — 8K→15K it rises +0.67 pp while the
baseline drops, and collapsed tasks recover (salad‑dressing 10→43 %, orange‑juice 17→33 %). It is
Object‑specific (Spatial/Long/Goal are fine under the same regime). Confidence: high it's real and
training‑driven; the clean separation (pure under‑training vs partial negative transfer) is a V2‑Object‑to‑30K
follow‑up. PH does not help here (it hurts in‑dist on 3 of 4 suites).

## Checkpoints
The V2 checkpoints (`runs/` 8K + `runs_ft15k/` 15K) were backed up to HF
**`AyushShah1107/act-deeponet-libero-checkpoints`** under `v2/` (see `backup_v2_to_hf.py`) and then
pruned locally. Only the eval logs + this summary are kept in‑repo.
