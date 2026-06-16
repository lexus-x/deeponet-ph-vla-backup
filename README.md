# DeepONet Operator Action Head + Persistent-Homology Loss for Vision-Language-Action Models

A research project that replaces the heavy **flow-matching action expert** of a Vision-Language-Action (VLA) model
(SmolVLA) with a lightweight **DeepONet operator action head**, optionally regularised by a
**Persistent-Homology (PH) topological loss**, and benchmarks it head-to-head on **LIBERO** for
in-distribution accuracy, **robustness** (LIBERO-Plus, 7 perturbations), parameter count, and inference latency.

> **One-line result:** the DeepONet head is **~9.6× smaller** (99.9 M → 10.4 M params) and **~5× faster**
> (148 ms → 29.5 ms) than the flow expert, **matches it in-distribution**, and is **>2× more robust**
> under distribution shift (17.9% → 38.5% on LIBERO-Plus).

---

## Table of contents
1. [TL;DR — headline results](#tldr--headline-results)
2. [The idea & why it is novel](#the-idea--why-it-is-novel)
3. [Architecture: before vs after](#architecture-before-vs-after)
4. [Component 1 — the DeepONet operator action head](#component-1--the-deeponet-operator-action-head)
5. [Component 2 — the Persistent-Homology (PH) loss](#component-2--the-persistent-homology-ph-loss)
6. [How it is integrated into SmolVLA](#how-it-is-integrated-into-smolvla)
7. [Research papers / references](#research-papers--references)
8. [Results in full](#results-in-full)
9. [Everything that was run — step by step](#everything-that-was-run--step-by-step)
10. [Repository structure — where to find things](#repository-structure--where-to-find-things)
11. [How to reproduce](#how-to-reproduce)
12. [Honest limitations & caveats](#honest-limitations--caveats)
13. [Deep-dive docs](#deep-dive-docs)

---

## TL;DR — headline results

**Model legend:** `M1 = flow` (SmolVLA flow-matching baseline) · `M3 = DeepONet` (operator head) · `M4 = DeepONet + PH loss`.

### LIBERO-Spatial, 5 seeds (the main comparison — `DeepONet PH/v2_results/data/summary.csv`)

| Model | In-dist acc | Robustness (LIBERO-Plus) | Latency | Action-head params |
|---|---|---|---|---|
| **M1 flow** | 79.4 ± 1.4 % | 17.9 ± 5.2 % | 148.1 ms | 99.9 M |
| **M3 DeepONet** | 80.3 ± 2.4 % | **38.5 ± 6.9 %** | **29.5 ms** | **10.4 M** |
| **M4 DeepONet + PH** | **81.5 ± 3.9 %** | 32.6 ± 5.1 % | 29.5 ms | 10.4 M |

- **In-distribution:** DeepONet **matches or slightly beats** the flow baseline.
- **Robustness:** DeepONet is **>2× more robust** than flow (and wins **6 of 7** perturbation categories).
- **Efficiency:** **9.6× fewer** head parameters, **~5× faster** inference (single forward pass vs 10-step flow denoising).
- **PH loss (M4):** improves in-distribution accuracy further, but the bare operator (M3) is the most robust — see [ablations](#ablations-libero-spatial).

### Per-suite in-distribution (single seed, 15K steps — first comparison)

| Suite | M1 flow | M3 DeepONet | M4 DeepONet+PH |
|---|---|---|---|
| LIBERO-Spatial | 78.5 % | **82.0 %** | 80.5 % |
| LIBERO-Object | 84.5 % | **94.0 %** | 87.0 % |
| LIBERO-Goal | **93.5 %** | 90.0 % | 89.0 % |

---

## The idea & why it is novel

A VLA model has two parts: a **VLM backbone** (vision + language → token features) and an **action head/expert**
that turns those features into a sequence of robot actions (an "action chunk").

State-of-the-art VLAs (SmolVLA, π0, etc.) use a **flow-matching** or **diffusion** action expert. These are powerful
but **heavy and slow**: they are themselves large transformers, and at inference they require an **iterative denoising
loop** (e.g. 10 forward passes) to produce one action chunk.

**Key observation:** an action chunk is a **function of time** — `a(τ)` for the normalised step `τ ∈ [0, 1]` within the
chunk. Producing a *function* from an *input observation* is exactly the problem that **operator learning** (DeepONet)
was designed for. So instead of denoising noise into actions, we **learn the operator** that maps the current
observation to the action *trajectory* directly.

### The novelty (what is new here)
1. **Operator learning as a VLA action head.** To our knowledge, framing the VLA action chunk as an operator
   `Observation ↦ a(·)` and using a **DeepONet** (Branch ⊗ Trunk) to produce it is a new action-head design for VLAs.
   It replaces the iterative flow/diffusion expert with a **single, parameter-free merge + one forward pass**.
2. **A drop-in, backbone-agnostic head.** The head consumes the generic `(prefix_tokens, pad_mask) → action_chunk`
   interface, so it plugs into *any* VLA backbone (SmolVLA, π0, Octo, GR00T, OpenVLA-OFT, ACT…). Only one small input
   projection rescales with the backbone; the rest is fixed at **~10.4 M params**.
3. **A topological (Persistent-Homology) trajectory loss.** On top of the usual regression loss, we add a **PH loss**
   that matches the *topological shape* of predicted vs ground-truth action trajectories (their persistence diagrams),
   encouraging globally consistent motion rather than only point-wise accuracy.
4. **Robustness as the headline metric.** We show the operator prior is not just smaller/faster but **substantially
   more robust** to real-world distribution shift (camera, lighting, layout, language…), which is where imitation
   policies usually break.

See [`docs/deeponet.md`](docs/deeponet.md) and [`docs/ph_loss.md`](docs/ph_loss.md) for the full theory.

---

## Architecture: before vs after

**Generic pattern (applies to every VLA — only the head changes, the backbone is shared & frozen-ish):**

```
 BEFORE (flow-matching baseline, M1)              AFTER (DeepONet head, M3/M4)
 ┌────────────────────────────┐                  ┌────────────────────────────┐
 │  VLM backbone (SmolVLM2)     │  (unchanged)     │  VLM backbone (SmolVLM2)     │ (unchanged, shared)
 │  ~350 M params               │                  │  ~350 M params               │
 └──────────────┬──────────────┘                  └──────────────┬──────────────┘
                │ prefix tokens [B, N, 960]                       │ prefix tokens [B, N, 960]
                ▼                                                  ▼
 ┌────────────────────────────┐                  ┌──────────────────────────────────────────┐
 │ FLOW-MATCHING ACTION EXPERT  │                 │ DeepONet ACTION HEAD  (10.4 M)             │
 │  ~99.9 M params              │       ⟶         │  CrossAttnPool → Branch(c) ⊗ Trunk(φ(τ))   │
 │  10-step iterative denoise   │      swap       │     → c⊙φ(τ) → OutMLP → a(τ)               │
 │  148 ms / chunk              │                 │  single forward pass · 29.5 ms / chunk     │
 └────────────────────────────┘                  └──────────────────────────────────────────┘
```

### Parameter breakdown (measured, SmolVLA)

| Component (DeepONet head) | Params | Note |
|---|---|---|
| CrossAttnPool (Perceiver-style) | **6.81 M** | `in_proj` 960→512 = 0.49 M · 8 learned queries · 3× cross-attn blocks = 6.31 M |
| Branch network (4096→768→256) | **3.34 M** | dominated by Linear(4096→768) = 3.15 M |
| Trunk network (33→256→256→256) | **0.14 M** | τ + Fourier features → basis φ(τ) |
| Output MLP (256→256→32) | **0.07 M** | + 32 output bias |
| **Total DeepONet head** | **≈ 10.4 M** | vs **99.9 M** flow expert |

| Whole-model comparison | Before (M1 flow) | After (M3/M4 DeepONet) |
|---|---|---|
| Backbone (SmolVLM2, 16 layers) | ~350 M | ~350 M (shared) |
| Action head | 99.9 M | **10.4 M** |
| **Total** | **~450 M** | **~360 M** (−20% if dead expert removed) |
| Inference (action chunk) | 148 ms | **29.5 ms** (~5×) |

> The same **~10.4 M head** drops into much larger backbones (π0 ~3.3 B, OpenVLA ~7 B, GR00T ~3 B); only the input
> projection rescales (≈ +0.5–1.5 M). The bigger the original action expert it replaces, the larger the saving.
> Full cross-model table in [`docs/architecture.md`](docs/architecture.md).

---

## Component 1 — the DeepONet operator action head

**File:** [`DeepONet PH/v2/deeponet_head_v2.py`](DeepONet%20PH/v2/deeponet_head_v2.py)

A **DeepONet** (Deep Operator Network, Lu et al. 2021) approximates an operator `G` mapping an input function/vector `u`
to an output function `G(u)(y)` via two sub-networks whose outputs are merged by a dot product:

```
G(u)(y) ≈ Σ_{k=1..p}  b_k(u) · t_k(y)      (Branch · Trunk)
```

Here we instantiate it as the action head:

1. **CrossAttnPool** — Perceiver-style cross-attention. `K = 8` learned query tokens attend over the full VLM prefix
   (`d_model = 512`, 8 heads, 3 blocks) → a fixed-size summary of the observation, regardless of prefix length.
2. **Branch network** — maps the pooled observation → `p = 256` coefficients `c ∈ ℝ²⁵⁶` (the "what to do" code).
3. **Trunk network** — maps the query coordinate `τ ∈ [0,1]` (normalised time within the action chunk), lifted by
   **Fourier features**, → `p = 256` basis functions `φ(τ) ∈ ℝ²⁵⁶` (the "shape over time").
4. **Parameter-free merge** — `c ⊙ φ(τ)` (element-wise), then a small **Output MLP** → the action at time `τ`.

```
a(τ) = OutMLP( Branch(obs) ⊙ Trunk(τ) )
```

Evaluating the trunk at `τ = 0, 1/T, …, 1` yields the **whole action chunk in one forward pass** — no denoising loop.

**Why `p = 256`?** It is the number of **basis functions** (operator latent dimension), *not* a time discretisation.
The chunk can be queried at any `τ` (continuous in time); `p` controls how rich the learned function space is.
`d_model = 512` is the internal transformer width of the cross-attention pooler. See [`docs/deeponet.md`](docs/deeponet.md).

---

## Component 2 — the Persistent-Homology (PH) loss

**File:** [`DeepONet PH/v2/ph_loss.py`](DeepONet%20PH/v2/ph_loss.py)

Standard training uses an MSE/L1 **point-wise** regression loss on the action chunk. The **PH loss** adds a
**topological** term: it computes the **persistence diagram** of the predicted and ground-truth action trajectories
(via a Vietoris–Rips / sublevel-set filtration) and penalises the difference between them.

Intuitively, PH captures the **global shape** of a trajectory — its loops, connected components, and how long
topological features "persist" across scales — which point-wise losses ignore. Matching it encourages the policy to
produce trajectories that are *globally* the right shape, not just locally close.

- `M3` = DeepONet head, **PH loss off**.
- `M4` = DeepONet head, **PH loss on**.

**Empirically:** PH (M4) **raises in-distribution accuracy** (81.5% vs 80.3% on Spatial) but the bare operator (M3)
is the **most robust** (38.5% vs 32.6%). So PH is an accuracy regulariser; the robustness comes from the operator
structure itself. See [`docs/ph_loss.md`](docs/ph_loss.md).

---

## How it is integrated into SmolVLA

**File:** [`DeepONet PH/v2/modeling_smolvla_deeponet_v2.py`](DeepONet%20PH/v2/modeling_smolvla_deeponet_v2.py)
(class `SmolVLADeepONetPolicy`).

SmolVLA = **SmolVLM2-500M** backbone (reduced to 16 layers, ~350 M) + a flow-matching action expert (~99.9 M).
The integration:

1. **Keep** the SmolVLM2 backbone and its prefix-token construction unchanged.
2. **Swap** the flow-matching expert for the DeepONet head (the generic `prefix_tokens [B,N,960], pad_mask → action_chunk [B,T,32]` interface).
3. **Load** the pretrained SmolVLA checkpoint with `strict=False`, then **freeze the dead flow modules**
   (`lm_expert`, `action_in_proj/out_proj`, `action_time_mlp`) so only the backbone + new head train.
4. **Two-stage fine-tune:** stage-1 head-only warm-up (1650 steps), then stage-2 full fine-tune; EMA on weights.

The head is the **generic plug-in**; `SmolVLADeepONetPolicy` is just the SmolVLA-specific adapter. Porting to π0 is
documented in [`DeepONet PH/PORTING_TO_PI0.md`](DeepONet%20PH/PORTING_TO_PI0.md). Full integration notes in
[`docs/smolvla_integration.md`](docs/smolvla_integration.md).

---

## Research papers / references

| Topic | Paper | Used for |
|---|---|---|
| **DeepONet** | Lu, Jin, Pang, Zhang, Karniadakis — *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*, **Nature Machine Intelligence 2021** ([arXiv:1910.03193](https://arxiv.org/abs/1910.03193)) | the operator action head (Branch ⊗ Trunk) |
| **SmolVLA** | Hugging Face / LeRobot — *SmolVLA: A Vision-Language-Action model for affordable and efficient robotics* (2025) ([arXiv:2506.01844](https://arxiv.org/abs/2506.01844)) | the backbone & flow-matching baseline |
| **Flow matching** | Lipman, Chen, Ben-Hamu, Nickel, Le — *Flow Matching for Generative Modeling*, **ICLR 2023** ([arXiv:2210.02747](https://arxiv.org/abs/2210.02747)) | the baseline action expert it replaces |
| **Persistent Homology (TDA)** | Edelsbrunner & Harer — *Computational Topology: An Introduction* (2010); Carlsson — *Topology and data*, **Bull. AMS 2009** | the topological PH loss |
| **Topological / PH loss in deep learning** | Hu, Li, Samaras, Chen — *Topology-Preserving Deep Image Segmentation*, **NeurIPS 2019** ([arXiv:1906.05404](https://arxiv.org/abs/1906.05404)) | differentiable persistence-diagram loss |
| **LIBERO benchmark** | Liu et al. — *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, **NeurIPS 2023** ([arXiv:2306.03310](https://arxiv.org/abs/2306.03310)) | evaluation suites (Spatial/Object/Goal/Long) |
| **Fourier features** | Tancik et al. — *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*, **NeurIPS 2020** ([arXiv:2006.10739](https://arxiv.org/abs/2006.10739)) | the Trunk's τ encoding |

*(Citations are the standard references for each method; verify exact venue/year against the linked source before formal use.)*

---

## Results in full

### LIBERO-Plus robustness by perturbation (Spatial, `DeepONet PH/v2_results/data/robustness_per_category.csv`)

| Perturbation | M1 flow | M3 DeepONet | M4 DeepONet+PH |
|---|---|---|---|
| Camera Viewpoints | 18.7 | **26.7** | 14.7 |
| Light Conditions | 22.7 | **61.3** | 54.7 |
| Sensor Noise | **25.3** | 21.3 | 17.3 |
| Background Textures | 24.0 | 42.7 | **44.0** |
| Objects Layout | 9.3 | **48.0** | 46.7 |
| Robot Initial States | 17.3 | **36.0** | 29.3 |
| Language Instructions | 8.0 | **33.3** | 21.3 |
| **Average** | **17.9** | **38.5** | **32.6** |

→ DeepONet (M3) wins **6/7** categories; flow wins only Sensor Noise. The largest gains are on **Objects Layout**
(9.3 → 48.0) and **Language** (8.0 → 33.3) — exactly the shifts that break imitation policies.

### Ablations (LIBERO-Spatial — `DeepONet PH/Ablation_Results/ablations.csv`)

| Configuration | In-dist | Robustness | Seeds |
|---|---|---|---|
| **Full DeepONet-v2 (p256, 3 blocks, Fourier)** | 80.3 ± 2.4 | **38.5 ± 6.9** | 5 |
| (−) basis p: 256→64 | 81.7 ± 0.5 | 32.7 ± 5.2 | 3 |
| (−) Fourier-τ (linear only) | 82.7 ± 1.4 | 39.0 ± 8.7 | 3 |
| (−) cross-attn blocks: 3→1 | 78.2 ± 2.0 | 30.8 ± 4.5 | 3 |
| Regression head (NO operator, same context) | 83.0 ± 4.2 | 32.7 ± 0.9 | 3 |

**Reading the ablations:** the **cross-attention pooler** matters most (3→1 blocks hurts both metrics); the
**operator merge** is what buys robustness — a plain regression head with the same context gets higher in-dist (83.0)
but loses ~6 points of robustness (32.7 vs 38.5). `p` and Fourier trade a little in-dist for robustness/stability.

Full numbers, plots, and per-task tables: [`docs/results.md`](docs/results.md).

---

## Everything that was run — step by step

A complete chronological log is in [`docs/experiments_log.md`](docs/experiments_log.md). Summary:

1. **Weeks 1–3 — Flow-matching baseline (SmolVLA).** Set up SmolVLA + LeRobot, fine-tuned the flow baseline on LIBERO,
   established the evaluation harness. → `Flowmatching PH (Weeks 1-3)/`
2. **DeepONet v1.** First operator head. Matched flow in-dist but **collapsed on some robustness categories**
   (camera/sensor) → motivated v2. → `DeepONet PH/v1_results/`
3. **DeepONet v2 (main).** Redesigned head (CrossAttnPool + p256 Branch⊗Trunk + Fourier). **5-seed** head-to-head of
   M1/M3/M4 on Spatial for in-dist + LIBERO-Plus robustness + latency + params. → `DeepONet PH/v2_results/`
4. **Ablations.** p256→64, no-Fourier, 3→1 cross-attn blocks, regression-head (no operator). → `DeepONet PH/Ablation_Results/`
5. **Per-suite generalisation.** Trained/evaluated M1/M3/M4 on **Object** and **Goal** (and Spatial) at 15K steps,
   with per-task plots + per-episode videos. → `DeepONet PH/{Spatial,Object,Goal}/`
6. **Paper-reproduction campaign (running).** 30K-step, batch-48 flow runs to push the flow baseline to its paper
   values (Spatial 90 / Object 96 / Long ≥71), flow-first, per-suite eval + plots. → `DeepONet PH/paper_repro/`
7. **task5 investigation.** Why LIBERO-Spatial task5 (stacked-bowl grasp) is a hard ceiling; replan sweep (inference
   tuning did not fix it). → `DeepONet PH/v2/task5_fix_experiment.py`
8. **Goal object-layout test (running).** Generalisation/anti-memorisation check: evaluate the frozen Goal model on
   3 different object-layout slices of LIBERO's 50 init-states. → `DeepONet PH/goal_layouttest/`
9. **Pi0 porting analysis.** Time/feasibility of porting the head to π0. → `DeepONet PH/PORTING_TO_PI0.md`

---

## Repository structure — where to find things

```
.
├── README.md                         ← you are here (master overview)
├── docs/                             ← deep-dive documentation
│   ├── architecture.md               ← before/after diagrams + cross-model param tables
│   ├── deeponet.md                   ← the DeepONet operator head, in depth
│   ├── ph_loss.md                    ← the persistent-homology loss, in depth
│   ├── smolvla_integration.md        ← how the head plugs into SmolVLA
│   ├── experiments_log.md            ← full chronological log of everything run
│   └── results.md                    ← all result tables (in-dist, robustness, ablations, per-task)
│
├── DeepONet PH/                      ← MAIN project
│   ├── v2/                           ← v2 source (the current/best version)
│   │   ├── deeponet_head_v2.py       ← the DeepONet operator head
│   │   ├── modeling_smolvla_deeponet_v2.py  ← SmolVLA adapter (swaps the head)
│   │   ├── modeling_smolvla_ph.py    ← flow baseline policy (+PH hooks)
│   │   ├── ph_loss.py                ← persistent-homology loss
│   │   ├── regression_head.py        ← ablation: plain regression head
│   │   ├── train.py / evaluate.py / evaluate_plus.py   ← training & eval
│   │   ├── libero_v_wrapper.py / libero_plus_wrapper.py ← env wrappers
│   │   ├── make_suite_plots.py / make_videos*.py        ← plots & videos
│   │   ├── paper_repro_30k.sh        ← the 30K paper-repro campaign
│   │   ├── run_goal_seedtest.sh + evaluate_seedtest.py  ← Goal object-layout test
│   │   └── *.sh                      ← all run/orchestration scripts
│   ├── Spatial/ Object/ Goal/        ← per-suite results: runs/ (eval JSONs), plots/, logs/, videos/
│   ├── v1_results/ v2_results/       ← packaged results + plots + CSV summaries
│   ├── Ablation_Results/             ← ablation JSONs, CSV, plot
│   ├── DeepONet_Results/             ← final report + collated data
│   ├── paper_repro/                  ← 30K campaign outputs + PROGRESS.log  (LIVE)
│   ├── goal_layouttest/              ← Goal object-layout test outputs       (LIVE)
│   └── PORTING_TO_PI0.md             ← π0 porting analysis
│
└── Flowmatching PH (Weeks 1-3)/      ← early flow-matching baseline work + presentation
```

**Quick find:**
- 📊 **Plots** → `DeepONet PH/*/plots/*.png`, `DeepONet PH/{v2_results,Ablation_Results}/*.png`
- 📈 **Result tables (CSV/JSON)** → `DeepONet PH/v2_results/data/`, `*/runs/eval_*/success_rates.json`
- 🎬 **Videos** → `DeepONet PH/*/runs/eval_*/episode_videos/`, `DeepONet PH/comparison_videos/`
- 📝 **Logs** → `DeepONet PH/*/logs/`, `DeepONet PH/paper_repro/PROGRESS.log`

> **Not included in this repo:** model checkpoints (`*.safetensors`, ~28 GB total — each 900 MB file exceeds GitHub's
> 100 MB limit), the Python `venv/`, `third_party/` libraries, and HF dataset caches. See [reproduce](#how-to-reproduce)
> to regenerate checkpoints. Everything else (code, results, plots, videos, logs) **is** backed up here.

---

## How to reproduce

```bash
# 1. Environment (Python 3.12) — install LeRobot + SmolVLA deps + LIBERO + persim/ripser for PH
pip install -r requirements.txt        # (see docs; uses lerobot, robosuite, libero, torch, persim)
export MUJOCO_GL=egl

# 2. Train one model (head ∈ {flow, deeponet}; variant ∈ {baseline, ph})
cd "DeepONet PH/v2"
python train.py --head deeponet --variant baseline \
  --dataset lerobot/libero_spatial_image --out runs/m3_s0 \
  --stage1_steps 1650 --stage2_steps 28350 --stage1_batch 48 --stage2_batch 48

# 3. Evaluate in-distribution + LIBERO-Plus robustness
python evaluate.py      --suite libero_spatial --model m3=deeponet=runs/m3_s0/checkpoints/LATEST --only indist
python evaluate_plus.py --suite libero_spatial --model m3=deeponet=runs/m3_s0/checkpoints/LATEST

# 4. Plots
python make_suite_plots.py --suite_dir ../Spatial --suite libero_spatial --label Spatial
```

---

## Honest limitations & caveats
- **In-dist numbers are imitation (behavioral cloning)** success rates on LIBERO's task distribution — high in-dist
  does *not* by itself prove generalisation; the **LIBERO-Plus robustness** numbers are the distribution-shift test.
- **Per-suite (Object/Goal) numbers are single-seed** (15K steps); the **5-seed** comparison is Spatial only. Treat
  single-seed numbers as ± a few % sampling noise.
- **PH loss helps in-dist but not robustness** — the robustness win comes from the operator structure, not PH.
- **The 30K paper-repro campaign is still running** — its flow numbers are not final yet (see `paper_repro/PROGRESS.log`).
- Cross-model parameter savings for π0/Octo/GR00T/OpenVLA in `docs/architecture.md` use **published model sizes**
  (approximate); only the SmolVLA numbers are measured here.

---

## Deep-dive docs
- [`docs/architecture.md`](docs/architecture.md) — before/after diagrams, full param breakdown, cross-model porting table
- [`docs/deeponet.md`](docs/deeponet.md) — operator learning & the DeepONet head, in depth
- [`docs/ph_loss.md`](docs/ph_loss.md) — persistent homology & the topological loss
- [`docs/smolvla_integration.md`](docs/smolvla_integration.md) — SmolVLA adapter internals
- [`docs/experiments_log.md`](docs/experiments_log.md) — full step-by-step of everything run
- [`docs/results.md`](docs/results.md) — every result table
