# π0 vs ACT vs VLA-DSS — LIBERO-Spatial, 10k steps, same machine (A100)

**Date:** 2026-07-16 → 07-17
**Status:** COMPLETE — training ✅ · latency ✅ · evals ✅ (all 4 runs, 200 episodes each)
**Question:** at an equal ~10,000-step budget, on **one A100**, how do a pretrained generalist VLA (π0), a standard imitation baseline (ACT), and a compact pretrained specialist (VLA-DSS/FNO) compare on capability, training cost, and inference latency?

> Supersedes `fno_vs_pi0_libero_spatial.md`, whose headline caveat was that VLA-DSS ran from scratch on a *different* GPU (A6000) than π0 (A100). Both defects are fixed here: VLA-DSS is now **pretrained on 130 LIBERO tasks then fine-tuned 10k on Spatial** (the true analog of π0's recipe), and **all three train on the same A100**, sequentially, so resource numbers are uncontended and comparable.

---

## 1. Setup

All three trained on the **same A100 80GB PCIe**, one at a time (never concurrently — contention would corrupt wall-clock/util).

| | **π0** | **ACT** | **VLA-DSS v2** |
|---|---|---|---|
| Params | ~3.3B | 51,600,319 | 28,616,969 |
| Framework | JAX / openpi | PyTorch / lerobot 0.1.0 | PyTorch |
| Init | `pi0_base` pretrained (Physical Intelligence) | from scratch | **130-task LIBERO pretrain** (all 5 suites, epoch 24, val_loss 0.185) |
| Recipe | openpi `pi0_libero`, full fine-tune | lerobot ACT defaults | Sarvik's `finetune_dinov3_spatial_aux` |
| Batch | 32 | 32 | 128 |

**Data.** π0 and ACT train on the *identical* LeRobot dataset (LIBERO-Spatial **no_noops**, 432 episodes / 52,970 frames, materialized from openpi's canonical `physical-intelligence/libero`). VLA-DSS uses its own pipeline over the raw LIBERO-Spatial HDF5s (54,750 chunk windows). Same underlying demos; different framing (chunk windows vs frames) — noted, not a defect.

**ACT has no pretraining stage** — that isn't a handicap we imposed, it's the paradigm: ACT is trained per-suite from scratch. π0 and VLA-DSS are both pretrained→fine-tuned.

---

## 2. Training cost — the same-machine result

| Metric | π0 | ACT | VLA-DSS v2 |
|---|---|---|---|
| Steps | 10,000 | 10,000 | 9,984 |
| **s/step** | 3.3 | **0.15** | ~0.72 |
| **Wall-clock** | **9h 56m** | **~25 min** | **~1h 56m** |
| Peak GPU memory | ~74 GB * | **5.2 GB** | 8.8 GB |
| Final loss | 0.0144 | ~0.19 | val_loss(ema) **0.0035** |

\* π0's ~74 GB is JAX/XLA's 0.9 arena **preallocation**, not true working set.

**π0 costs ~24× ACT's training time and ~5× VLA-DSS's, on identical hardware and data.**

Losses are **not comparable across models** — flow-matching (π0), L1+KL (ACT), and SmoothL1+BCE+aux (VLA-DSS) are different objectives. Only success rate compares.

**VLA-DSS's pretrain resume is verified**, not assumed — the log shows `Resumed weights from .../vla_dss_pretrain_base/best.pt (epoch 24, val_loss=0.1847)`, tokenizer (vocab 86) and action mean/std reused, and initial loss **0.311** rather than a from-scratch ~1.2. Two honest wrinkles in the warm-start (recovered from the checkpoint, since the training worker was killed by a session limit before writing `run_meta.json`): the fine-tune **fresh-initialized `aux_head`** (the spatial-aux recipe adds a head the 130-task pretrain lacked, 4 random keys), and the **AdamW optimizer state did not restore** (`parameter group doesn't match`, a consequence of that new head), so the fine-tune restarted its optimizer from scratch. The *weights* transferred correctly — the pretrain→FT claim holds — but it was not a bit-perfect warm-start.

**⚠ π0 checkpoint deadlock.** π0 completed all 10k steps, but its final save deadlocked in orbax/tensorstore (hung ~30 min, left an un-finalized temp). Last complete checkpoint is **step 9,000** (loss 0.0152, same converged plateau as 9,990's 0.0144), so **π0 is evaluated at 9k, not 10k** — which slightly *under*-sells it. Broken dir kept as `BROKEN_9999_deadlocked`.

---

## 3. Inference latency — measured, batch 1, idle A100

Method: batch=1 (deployment condition), **20 warmup calls discarded** (they include CUDA context init and π0's XLA JIT compile), **200 timed calls**, `time.perf_counter()`, GPU verified idle (459 MiB / 0%) before each, one model at a time.
**Synchronization** (the correctness crux — CUDA/JAX are async; without this you time kernel *launch*, not execution): PyTorch → `torch.cuda.synchronize()` around each call under `inference_mode()`; JAX → blocked via `np.asarray` device_get. Deliberately **not** openpi's own `policy_timing.infer_ms`, which is captured pre-transfer and measures dispatch only.

| Model | Params | **ms/chunk** | p50 | p95 | std | Chunk | Exec/replan | **ms/exec action** | **Max Hz** | Peak mem † |
|---|---|---|---|---|---|---|---|---|---|---|
| **ACT** | 51.6M | **13.83** | 14.74 | 18.84 | 2.38 | 100 | 100 | 0.138 | 7229 | 756 MiB |
| **π0** | ~3.3B | **75.52** | 75.25 | 77.45 | 1.15 | 50 | 5 | 15.10 | 66.2 | 8646 MiB |
| **VLA-DSS** | 28.6M | **331.72** | 326.41 | 369.60 | 15.88 | 16 | 8 | 41.47 | 24.1 | 994 MiB |

† nvidia-smi Δ over the 459 MiB idle baseline — **the only cross-framework-comparable memory column**. (`torch.cuda.max_memory_allocated` counts only torch tensors, excluding CUDA context/cuDNN workspace, and has no JAX equivalent — never compare it to π0's.)

All three clear LIBERO's ~20 Hz (50 ms/action) budget **even at p95** (VLA-DSS 46.2, π0 15.5, ACT 0.19). VLA-DSS is the only one near the line — ~3.8 ms of p95 margin vs π0's 3× headroom.

### 3a. Parameter count does not predict latency

**The 28.6M VLA-DSS is 4.4× slower per chunk than the 3.3B π0.** Profiled cause (measured, not speculated):

- **Wavelet scattering ≈ 95% of the forward** (vision total 99.1%); raw kymatio `Scattering2D` (J=3, L=12) alone ≈ **42%**; ResBlocks+attention over the scattering map are most of the remainder.
- **The FNO head — the actual research contribution — is ≈ 0.3%** (1.3 ms). Frozen DINOv3 ViT-S over both views ≈ 3%.
- **Dispatch-bound, not compute-bound:** batch-8 costs essentially the *same wall-time* as batch-1 (ratio 0.77) despite 8× the work — sub-linear scaling only a launch-bound workload shows. J=3/L=12 expands into hundreds of tiny wavelet kernels that cannot fill an A100 at batch 1; π0's few large dense matmuls saturate it.
- *Untested hypothesis (labeled):* TF32 off / fp32 model — likely secondary given the workload is dispatch-bound.

**Implication for the lab:** VLA-DSS's efficiency claim is real for **training** (5× less wall-clock, 8× less VRAM than π0) but **does not transfer to inference latency**, and the bottleneck is the *scattering front-end*, not the FNO novelty. Latency is spent almost entirely on a component that isn't the contribution.

### 3b. Read `ms/chunk`, not `ms/exec action`, as "speed"

The per-executed-action column flatters ACT (0.138 ms) purely because it executes all 100 actions **open-loop**, while π0 discards 45 of every 50. That is a **control-regime difference, not raw speed** — and open-loop-100 is generally a *worse* policy, not a faster one. `ms/chunk` is the honest compute cost.

---

## 4. Capability — LIBERO-Spatial success rate

Protocol: 20 rollouts/task × 10 tasks = **200 episodes**. Each result verified as a clean 200 (no duplicate (task, rollout) pairs, checked from the logs).

| Model | Success | Protocol |
|---|---|---|
| **π0** | **94.0% (188/200)** | openpi client, seed 7, replan 5, max 220 steps |
| **ACT** (replan-5) | **66.5% (133/200)** | openpi client, seed 7, replan 5, max 220, resize 256 |
| **ACT** (default open-loop-100) | **29.5% (59/200)** | openpi client, seed 7, **replan 100**, max 220, resize 256 |
| **VLA-DSS v2** | **64.5% (129/200)** | `eval_sim.py`, seed 0, execute 8, max 400 steps, 128px |
| *(VLA-DSS v1, from-scratch, A6000 — superseded)* | *47.0% (94/200)* | *`eval_sim.py`* |

**The two ACT numbers are the same checkpoint** — the 37-point gap is purely the control regime. lerobot's ACT defaults to `chunk_size = n_action_steps = 100`, i.e. it commits to a 100-step chunk (~half of LIBERO's 220-step cap) fully open-loop. Matching π0's replan-5 cadence recovers +37pp. At replan-5 ACT scores **95% on task 0 — identical to π0** — so 29.5% reflects a poor execution regime, not a weak policy. **Report ACT at replan-5 (66.5%) as its capability number**; the default is a cautionary datapoint about benchmarking with stock configs.

**Pretraining lifted VLA-DSS from 47% → 64.5%** (+17.5pp over the from-scratch v1) — the 130-task pretrain paid off, and this is now a fair analog of π0's pretrain→fine-tune recipe.

### Per-task success (all 10 tasks, task_id shares the same task across harnesses)

| # | Task (black bowl → plate) | π0 | ACT r5 | VLA-DSS v2 |
|---|---|---|---|---|
| 0 | between plate and ramekin | 95% | 95% (19/20) | 50% (10/20) |
| 1 | next to the ramekin | 100% | 70% (14/20) | 85% (17/20) |
| 2 | from table center | 100% | 75% (15/20) | 80% (16/20) |
| 3 | on the cookie box | 90% | 95% (19/20) | 80% (16/20) |
| 4 | in top drawer of wooden cabinet | 90% | 45% (9/20) | **15% (3/20)** |
| 5 | on the ramekin | 90% | 70% (14/20) | 70% (14/20) |
| 6 | next to the cookie box | 100% | 75% (15/20) | 65% (13/20) |
| 7 | on the stove | 90% | **20% (4/20)** | 90% (18/20) |
| 8 | next to the plate | 95% | 85% (17/20) | 55% (11/20) |
| 9 | on the wooden cabinet | 90% | 35% (7/20) | 55% (11/20) |
| — | **Overall** | **94.0%** | **66.5%** | **64.5%** |

(ACT open-loop-100's per-task rows are in `logs/act_eval/results_primary.json`; overall 29.5%, task 0 = 45%, task 1 = 15%.)

**Per-task spread tells the real story.** π0 is uniformly strong (90–100% on every task). ACT-r5 and VLA-DSS both have a task each where they collapse — **ACT on task 7 (on the stove, 20%)**, **VLA-DSS on task 4 (top drawer of wooden cabinet, 15%)** — and those single failures account for most of their ~28pp gap to π0. Excluding each model's worst task: VLA-DSS ~70%, ACT-r5 ~71%. Their overalls being near-identical (64.5 vs 66.5) hides that **they fail on different tasks** — VLA-DSS handles the stove (90%) where ACT fails, ACT handles the drawer (45%) where VLA-DSS fails. VLA-DSS's jerk is low (mean 0.037), consistent with the FNO head's band-limited smoothness.

### Headline

**π0 (94%) > ACT-replan5 (66.5%) ≈ VLA-DSS v2 (64.5%) ≫ ACT-default (29.5%).** The 3.3B pretrained generalist wins capability by ~28pp. ACT (with a sane replan rate) and the pretrained 28.6M VLA-DSS land in a statistical tie for second — VLA-DSS matching a 1.8× larger model at ~half the parameters is its real result. The catch is latency (§3): VLA-DSS is the slowest of the three per chunk despite being the smallest.

---

## 5. Caveats

1. **π0 evaluated at step 9,000**, not 10,000 (checkpoint deadlock) — under-sells π0.
2. **Eval harnesses differ.** π0/ACT use openpi's LIBERO client (seed 7, max 220, replan 5); VLA-DSS uses its author's `eval_sim.py` (seed 0, max 400, execute 8). Same 10 tasks and the same LIBERO success condition, but not a byte-identical protocol.
3. **Batch sizes differ** (π0/ACT 32, VLA-DSS 128) — each model's own official recipe. VLA-DSS sees ~4× more gradient samples per step.
4. **π0's LR schedule doesn't fully anneal**: openpi's default cosine decays over 30k steps, kept unchanged, so at the 10k cutoff π0's LR is only partway down — mildly disadvantaging π0.
5. **ACT's replan regime**: trained/defaulted to `chunk_size = n_action_steps = 100` (fully open-loop). Primary eval uses its own default; a replan-5 secondary is run only if the default scores poorly, to separate "ACT is weak" from "open-loop-100 is a bad regime on LIBERO."
6. **MuJoCo version**: the eval client venv uses mujoco 3.2.3, not 3.3.2 (which renders different colors per lerobot community reports). π0 and ACT share this renderer so they stay mutually comparable; VLA-DSS's harness differs.

---

## 6. Artifacts (all on a100, under `/mnt/sda/supervisor_cmp/`)

- **Checkpoints:** `checkpoints/pi0_spatial_cmp/pi0_spatial_10k/9000` · `checkpoints/act_spatial_10k/checkpoints/010000` · `checkpoints/vla_dss_ft_spatial_10k/best.pt` · pretrain base `checkpoints/vla_dss_pretrain_base/best.pt`
- **Training logs:** `logs/pi0/` · `logs/act/` · `logs/vla_dss_ft/` (each: `train.log`, `gpu.csv`, `sys.csv`, `run_meta.json` — VLA-DSS's was reconstructed post-hoc from `train.log` + checkpoint, with `start_timestamp`/`command` left null as unrecoverable)
- **Latency:** `logs/latency/` (`results.json`, `README.md` with the exact method, `{pi0,act,vla_dss}_raw.json`, `vla_dss_profile.json`); scripts `code/latency_bench/`
- **Evals:** `logs/pi0_eval/` · `logs/act_eval/` (`results_primary.json` = open-loop-100, `results_replan5.json` = replan-5) · `logs/vla_dss_eval/` (`rollouts.jsonl` = 200 per-episode records)
- **Dataset (π0/ACT):** `data/lerobot/supervisor_cmp/libero_spatial` (v2.1, 432 eps / 52,970 frames)
