# ACT + DeepONet Operator Action Head (+ Persistent-Homology loss) — LIBERO

This is the **ACT-backbone** companion to the SmolVLA study in the parent repo. It replaces ACT's
**transformer action decoder** with a lightweight **DeepONet operator action head** (optionally
regularised by a **gated Persistent-Homology (PH) topological loss**) and benchmarks all three heads
**in-distribution on the four LIBERO suites** (Spatial / Object / Goal / Long).

> **One-line result:** the DeepONet head **matches or beats** the ACT decoder in-distribution on **all four
> suites** while using a **3.7× smaller action head** (37.8 M → 10.2 M params; 88.3 M → 60.7 M total). The PH
> regulariser does **not** help in-distribution accuracy (it lowers it on every suite).

> **Scope note:** this repository contains the **architecture, training/eval code, and in-distribution
> results**. The LIBERO-Plus **robustness results** and the third-party LIBERO-Plus benchmark data/install are
> kept out of this repo by design (the in-distribution study is self-contained).

---

## Architecture

```
                              INPUT  (LIBERO observation, fps 10)
  ┌────────────────┬────────────────┬───────────────┬────────────────────────────┐
  │ agentview img  │   wrist img    │  robot state  │   language instruction      │
  │   3×256×256    │   3×256×256    │     8-dim     │  "pick up the black bowl…"  │
  └───────┬────────┴───────┬────────┴───────┬───────┴──────────────┬─────────────┘
          │                │                │                      │
   ┌──────▼────────────────▼──────┐   ┌─────▼──────┐    ┌───────────▼────────────┐
   │     ResNet-18 backbone       │   │ state proj │    │  TinyLanguageEncoder    │
   │     (shared, 11.2 M)         │   │            │    │  BERT tokenizer + 2-layer│
   │       → visual tokens        │   │            │    │  transformer  (~4 M)     │
   └──────────────┬───────────────┘   └─────┬──────┘    └───────────┬─────────────┘
                  └─────────────────┬───────┴───────────────────────┘
                                    │  tokens  (+ CVAE style latent z, 32-d, train-only)
                          ┌─────────▼──────────┐
                          │ Transformer encoder │   4 layers · d_model 512 · ff 3200
                          └─────────┬──────────┘
                                    │  memory  [B, N, 512]
              ┌─────────────────────┴─────────────────────────┐
              │                                                │
     ══ ACT baseline ══                       ══ ACT + DeepONet  (+PH) ══
  ┌────────────────────────┐         ┌────────────────────────────────────────────┐
  │  Transformer decoder   │         │  CrossAttnPool : 8 learned queries,         │
  │  7 layers  (37.8 M)    │         │    Perceiver-style cross-attention × 3 blocks│
  │                        │         │                     │                        │
  │                        │         │            ┌────────┴─────────┐              │
  │                        │         │       Branch b(u)          Trunk φ(τ)        │
  │                        │         │       p = 256 coeffs       Fourier τ, n = 6  │
  │                        │         │            └────────┬─────────┘              │
  │                        │         │      Hadamard  c ⊙ φ  →  out_mlp(·) + bias   │
  │                        │         │            DeepONet head  (10.2 M)           │
  └───────────┬────────────┘         └───────────────────────┬─────────────────────┘
              │                                               │
              ▼                                               ▼
     action chunk [T = 100, 7-DoF]                  action chunk [T = 100, 7-DoF]

  Loss:  L1 (action reconstruction) + KL (CVAE)
         + gated Persistent-Homology loss   (ACT+DeepONet+PH only)
           PH: warm-up 5000 steps · per-sample trigger 0.15 · λ = 0.02 · 0 params (loss-only)

  Params:   ACT 88.3 M   →   ACT+DeepONet 60.7 M      (action head 37.8 M → 10.2 M, 3.7× smaller)
```

**Why a DeepONet head?** A DeepONet (Lu et al., *Nature Machine Intelligence* 2021) approximates an *operator*
`G: u ↦ G(u)(τ)` as a **branch** network (encodes the conditioning `u` into `p` coefficients) times a **trunk**
network (a `τ`-indexed basis). Predicting an action *chunk* (a function over the chunk index `τ`) from the
current observation is exactly an operator-learning problem, so the head produces the whole chunk in a single
forward pass — no autoregression, no diffusion/flow denoising — at a fraction of the decoder's parameters.

**The PH loss** matches the *topological shape* (persistence diagram) of predicted vs ground-truth action
trajectories. It is **gated** (only fires after a warm-up and on samples above an error trigger) and adds **no
parameters**. In this study it did not improve in-distribution accuracy.

---

## In-distribution results (LIBERO, success rate %)

Each number is mean success over the suite's 10 tasks (re-evaluation run, replan 5). Higher is better.

| Suite | ACT (baseline) | ACT + DeepONet | ACT + DeepONet + PH |
|---|---|---|---|
| LIBERO-Spatial | 83.7 | 83.7 | 78.0 |
| LIBERO-Object | 81.7 | **87.0** | 72.0 |
| LIBERO-Long (LIBERO-10) | 54.7 | **62.7** | 50.0 |
| LIBERO-Goal | 83.3 | **86.3** | 85.0 |
| **Average** | 75.9 | **79.9** | 71.3 |

- **ACT + DeepONet ≥ ACT baseline in-distribution on all four suites** (tie on Spatial), at a **3.7× smaller
  action head** and ~31% fewer total parameters.
- **PH does not help in-distribution** — it lowers accuracy on every suite; the bare operator head is the one to use.

---

## Components & files

| File | What it is |
|---|---|
| [`deeponet_head_v2.py`](deeponet_head_v2.py) | DeepONet head: CrossAttnPool → branch ⊗ trunk (Hadamard) → out_mlp |
| [`modeling_act_deeponet.py`](modeling_act_deeponet.py) | ACT policy with the decoder swapped for the DeepONet head + language token |
| [`ph_loss_gated.py`](ph_loss_gated.py) / [`ph_loss.py`](ph_loss.py) | gated persistent-homology topological loss |
| [`lang_encoder.py`](lang_encoder.py) | `TinyLanguageEncoder` (BERT tokenizer + 2-layer transformer, ~4 M) |
| [`train_act.py`](train_act.py) | training (30K steps, batch 64, EMA, RAM-cached dataloader, optional augmentation) |
| [`evaluate_act.py`](evaluate_act.py) | closed-loop in-distribution evaluation on LIBERO |
| [`ram_cache.py`](ram_cache.py) | decode-once RAM cache (eliminates per-epoch image re-decode) |
| [`act_common.py`](act_common.py) | policy builder / checkpoint I/O |
| `run_act_campaign.sh` | the full 4-suite × 3-variant campaign |

Results live under [`act_results/<Suite>/runs/`](act_results/) (per-variant training logs, in-distribution
`success_rates.json`, per-task plots, and rollout videos). Model checkpoints are **not** committed (see `.gitignore`).

---

## Reproduce

```bash
source ../venv/bin/activate
export MUJOCO_GL=egl
# train one variant on one suite
python train_act.py --variant act_deeponet --dataset lerobot/libero_object_image \
  --out act_results/Object/runs/act_deeponet --steps 30000 --batch 64 --ckpt_every 5000
# in-distribution evaluation (3 seeds)
python evaluate_act.py --model act_deeponet=act_results/Object/runs/act_deeponet/checkpoints/LATEST \
  --suite libero_object --dataset lerobot/libero_object_image --out act_results/Object/runs/eval_indist \
  --indist_episodes 10 --test_seeds 3 --replan 5 --only indist --max_steps 520
```

Variants: `act` (baseline) · `act_deeponet` (operator head) · `act_deeponet_ph` (operator head + PH loss).

---

## Honest notes
- Numbers above are **in-distribution** behavioral-cloning success rates (20 ep/task on the re-eval; 10 ep × 3
  seeds protocol available via `evaluate_act.py`). High in-distribution success does not by itself prove robustness.
- The DeepONet head's defensible in-distribution claim is **parity at lower cost**, not a clear accuracy win.
- **PH is not recommended** (no in-distribution benefit here).
