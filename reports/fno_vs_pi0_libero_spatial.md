# Fresh VLA-DSS (FNO) vs. Pretrained π0 — LIBERO-Spatial, 10k-step budget

**Date:** 2026-07-16 → 07-17
**Question:** How far does a *from-scratch*, tiny, RGB-only policy (Sarvik's VLA-DSS / FNO-VLA) get against a *pretrained*, large generalist VLA (Physical Intelligence π0, full fine-tune) when both are given the **same ~10,000-step training budget** on the **same suite (LIBERO-Spatial)**?

---

## TL;DR

| | **Fresh VLA-DSS (FNO)** | **Pretrained π0** |
|---|---|---|
| **LIBERO-Spatial success** | **47.0%** (94/200) | **94.0%** (188/200) |
| Params | 28.6M total (~7M trainable) | ~3.3B (full fine-tune) |
| Init | from scratch | π0_base pretrained |
| GPU | RTX A6000 48GB | A100 80GB |
| Train wall-clock | **~3h 38m** | ~9h 56m |
| Train peak VRAM | **8.5 GB** | ~74 GB (JAX 0.9 arena) |
| Steps × batch | 9,984 × 128 | 10,000 × 32 |

**Verdict:** π0 wins decisively on capability (2× the success rate, wins every one of the 10 tasks). VLA-DSS's story is efficiency, not accuracy — it reaches ~half of π0's success with **~115× fewer parameters, ~1/8th the VRAM, and ~2.7× less wall-clock on a cheaper GPU**. This is the expected outcome of pretrained-large vs scratch-small at an equal, short step budget; it is not evidence either way about VLA-DSS's *published* recipe (see caveats).

---

## 1. Setup

Both runs launched in parallel on separate boxes (so time/util are on different hardware — see caveat 4). Everything written under fresh `supervisor_cmp` workspaces; no intern data touched.

### VLA-DSS (FNO-VLA)
- **Architecture:** wavelet-scattering + **frozen DINOv3 ViT-S** vision, FiLM fusion, **Fourier-Neural-Operator** action head, RGB-only, train-only aux-xy grasp head. (Sarvik's `SarvikIIT/VLA-DSS`.)
- **Params:** 28.6M total (vision 22.9M incl. frozen DINOv3 21.6M; fusion 2.8M; FNO 2.4M; language 0.4M). ~7M trainable — the frozen backbone is architectural, kept as designed; everything else random-init ("fresh").
- **Config:** his published-best spatial recipe (aux-xy on, `aux_xy_weight 0.5`) with from-scratch hyperparams (lr 2e-4, 1k warmup, wd 1e-5, batch 128, EMA 0.999). `data.suite: libero_spatial`, img 128, random-crop + color-jitter aug.
- **Data:** LIBERO-Spatial, 10 tasks × 50 demos → 54,750 chunk windows (90/10 train/val split).
- **Box:** RTX A6000 48GB.

### π0
- **Architecture:** π0 (PaliGemma VLM + flow-matching action expert), **full fine-tune** from `gs://openpi-assets .../pi0_base` (not LoRA).
- **Config:** openpi `pi0_libero` recipe unchanged except: spatial-only dataset, `num_train_steps=10000`, batch 32, wandb off. LR schedule left at openpi default (cosine, warmup 1000, peak 2.5e-5, `decay_steps=30000`).
- **Data:** LIBERO-Spatial **no_noops**, 432 episodes / 52,970 frames / 10 tasks, materialized to LeRobot format from openpi's canonical `physical-intelligence/libero` (episodes 1261–1692, task_index 30–39). Identity verified 4 ways.
- **Box:** A100 80GB.

---

## 2. Training — resource & convergence

| Metric | VLA-DSS | π0 |
|---|---|---|
| Total steps | 9,984 (26 epochs) | 10,000 (reached) |
| Batch size | 128 | 32 |
| Gradient samples seen | ~1.28M | ~0.32M |
| Throughput | 1.22 s/step | 3.3 s/step |
| Wall-clock | ~3h 38m (14:17→17:56) | ~9h 56m |
| Peak GPU memory | **8,480 MiB** | ~73,963 MiB (XLA 0.9 preallocation; true working set lower) |
| GPU util | 64–100% | 100% |
| GPU power | ~262 W peak | 260–324 W |
| GPU temp | 85 °C | 60–66 °C |
| Final train loss | 0.0015 (val_loss ema **0.0023**) | ~0.0144 |

Both converged cleanly (VLA-DSS SmoothL1→~0.002; π0 flow-matching loss 0.147→0.014).

**⚠ π0 checkpoint deadlock:** π0 completed all 10,000 steps, but the **final checkpoint save deadlocked** in orbax/tensorstore (`wait_until_finished()` hung ~30 min on a stuck `ts_pool_worker`, leaving an un-finalized temp dir). The last *complete* checkpoint is **step 9,000** (loss 0.0152, on the same converged plateau as 9,990's 0.0144). **π0 was therefore evaluated at step 9,000, not 10,000** — a ~10% step shortfall that if anything *under*-sells π0. Broken dir preserved as `BROKEN_9999_deadlocked`.

---

## 3. Evaluation — LIBERO-Spatial success rate

Both: 20 rollouts/task × 10 tasks = **200 episodes**, on the same 10 spatial tasks and the same LIBERO built-in success condition.

### Per-task (task_id maps to the same task in both harnesses)

| # | Task (black bowl → plate) | VLA-DSS | π0 | Δ |
|---|---|---|---|---|
| 0 | between plate and ramekin | 55% (11/20) | 95% (19/20) | +40 |
| 1 | next to the ramekin | 35% (7/20) | 100% (20/20) | +65 |
| 2 | from table center | 55% (11/20) | 100% (20/20) | +45 |
| 3 | on the cookie box | 80% (16/20) | 90% (18/20) | +10 |
| 4 | in top drawer of wooden cabinet | 50% (10/20) | 90% (18/20) | +40 |
| 5 | on the ramekin | 70% (14/20) | 90% (18/20) | +20 |
| 6 | next to the cookie box | 20% (4/20) | 100% (20/20) | +80 |
| 7 | on the stove | 45% (9/20) | 90% (18/20) | +45 |
| 8 | next to the plate | 40% (8/20) | 95% (19/20) | +55 |
| 9 | on the wooden cabinet | 20% (4/20) | 90% (18/20) | +70 |
| — | **Overall** | **47.0% (94/200)** | **94.0% (188/200)** | **+47** |

π0 wins every task. VLA-DSS is competitive only on task 3 (80 vs 90); its worst tasks (6, 9 at 20%) are π0's near-perfect ones (100/90) — the largest gaps.

### Eval resource use

| Metric | VLA-DSS | π0 |
|---|---|---|
| Eval wall-clock | 3h 18m | **29 min** |
| Peak GPU memory | 2,172 MiB | 62,839 MiB |
| Avg GPU util | 4.6% | 15.8% |
| Avg power | 82 W | 93 W |
| Mean steps-to-success | 137.5 | not logged by harness |

Both evals are CPU/physics-bound (MuJoCo). VLA-DSS's eval was ~7× slower in wall-clock **only because its harness runs robosuite with numba JIT disabled** (to dodge a mujoco-binding bug) — pure-Python physics. This is an artifact of the eval harness, **not** a property of the model, and should not be read as π0 being "faster to evaluate."

---

## 4. Fairness caveats (read before quoting these numbers)

1. **This is pretrained-large vs scratch-small by design.** π0 starts from a checkpoint trained on huge robot data; VLA-DSS starts from noise. The 94 vs 47 gap is mostly the pretraining, not the architecture.
2. **VLA-DSS's published spatial number is ~73%** with its *full* recipe (~80-epoch cross-suite pretrain + fine-tune). The 47% here is a deliberately handicapped *from-scratch, 10k-step* run — not VLA-DSS at its best.
3. **Different eval harnesses.** Same 10 tasks and same LIBERO success check, but different caps: VLA-DSS `eval_sim.py` (max 400 steps, execute-8, seed 0) vs openpi `main.py` (max 220 steps, replan-5, `num_steps_wait=10`, resize 224, seed 7). Not a byte-identical protocol.
4. **Different hardware** (A6000 vs A100) — chosen for parallel throughput — so training wall-clock/util/power are indicative, not a controlled same-GPU comparison.
5. **Different batch sizes** (128 vs 32), each the model's own official recipe. VLA-DSS saw ~4× more gradient samples at equal step count — the intended "make it a bit easier for VLA-DSS" lever, applied through legitimate per-model configs.
6. **π0 evaluated at step 9,000** (checkpoint deadlock), not 10,000 — under-sells π0 slightly.

---

## 5. Artifacts & provenance

**VLA-DSS (a6000, Windows):**
- Checkpoint: `D:\supervisor_cmp\checkpoints\fresh_spatial_10k\best.pt` (val_loss 0.0023)
- Train logs: `D:\supervisor_cmp\logs\` (`train.log`, `gpu.csv`, `sys.csv`, `run_meta.json`)
- Eval: `D:\supervisor_cmp\logs\eval_vla\` (`results.json`, `rollouts.jsonl` (200 records), `eval.log`, `gpu.csv`)

**π0 (a100, Linux):**
- Checkpoint evaluated: `/mnt/sda/supervisor_cmp/checkpoints/pi0_spatial_cmp/pi0_spatial_10k/9000`
- Broken final: `.../BROKEN_9999_deadlocked`
- Train logs: `/mnt/sda/supervisor_cmp/logs/pi0/` (`train.log`, `gpu.csv`, `sys.csv`, `run_meta.json`)
- Eval: `/mnt/sda/supervisor_cmp/logs/pi0_eval/` (`results.json`, `eval.log`, `server.log`, `gpu.csv`, `videos/`)
- Dataset: `/mnt/sda/supervisor_cmp/data/lerobot/supervisor_cmp/libero_spatial`

*Bottom line: at an equal 10k-step budget, pretrained π0 roughly doubles fresh VLA-DSS's LIBERO-Spatial success (94% vs 47%) and wins every task, while VLA-DSS delivers ~half the capability at a tiny fraction of the parameters, memory, and training time. The interesting comparison for the lab is not "who wins" (pretraining wins) but the efficiency frontier VLA-DSS holds at 115× smaller scale.*
