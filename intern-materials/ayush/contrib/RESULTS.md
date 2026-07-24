# DeepONet Contributions — Implementation + Results (2026-07-17)

Code, self-checks, and mechanism experiments for the three proposed contributions to
Ayush's `DeepONet-PH-VLA`. Everything here ran on real data where possible
(**3,379 real LIBERO-Spatial action chunks**, 50×7, extracted from the exact training
dataset `lerobot/libero_spatial_image` — parquet only, no videos).

**What this is / isn't:** these are *mechanism-level* results + drop-in modules,
runnable on CPU. The LIBERO **success-rate** numbers require the lab GPU + the ~28GB
checkpoints (not in the repo) — the modules below plug into `train.py`/`evaluate.py`
for those runs.

## Files

| File | What | Status |
| --- | --- | --- |
| `operator_chunk_exec.py` | Contribution #1: continuous-τ query with cached code, fractional-τ ensembling, exact boundary pinning, analytic da/dτ replan points | all 5 self-checks pass (incl. `query(grid) == forward()` vs the real `DeepONetHeadV2`) |
| `pod_trunk.py` | Contribution #2: POD "eigen-motion" trunk, drop-in head with same interface | all 5 self-checks pass |
| `ph_loss_stable.py` | Contribution #3: TRUE H0 persistence (batched differentiable MST) + PCA projection | all 5 self-checks pass — spectrum **exactly matches ripser's H0 deaths** |
| `run_experiments.py` | Experiments A/B/C below | `results/results_raw.json`, `results/expC_eigenmotions.png` |
| `fetch_libero_actions.py` | Pulls action parquets (no videos) → `data/chunks_spatial.npy` | 3,379 chunks |

## Exp A — chunk-execution strategies (closed-loop simulation, 20 seeds)

Simulated H=50 chunks, replan every 25, systematic per-replan prediction error:

| strategy | boundary jump | tracking RMSE | jerk |
| --- | --- | --- | --- |
| naive switching | 0.384 ± 0.038 | **0.109 ± 0.003** | 0.0067 |
| fractional-τ ensembling | 0.375 ± 0.037 | **0.107 ± 0.003** | 0.0066 |
| boundary pinning | **0.000** | 0.122 ± 0.005 | **0.0049** |
| ensemble + pinning | **0.000** | 0.120 ± 0.005 | **0.0048** |

**Read:** pinning eliminates boundary discontinuity *exactly* and cuts jerk ~27%, at the
cost of +12% tracking RMSE during the decay window (the offset drags the trajectory).
Ensembling alone is marginal when per-replan errors are systematic (they don't average
out). **Honest implication:** pinning is the right tool where discontinuities cause
failures (jerk-sensitive grasps, chunk-boundary drops on LIBERO-Long); it is not a free
accuracy win — evaluate on Long with the boundary-artifact metric first
(`log |a_new(0) − a_old(τ_switch)|` on success vs failure episodes).

## Exp B — topological-loss stability (real chunks, padded ℝ³², 300 pairs)

Can each loss tell "same trajectory, perturbed" (should be cheap) from "different
trajectory" (should be expensive)? And does it tolerate a mild time-warp (same motion)?

| loss | AUC (same < diff, →1 better) | warp/diff ratio (→0 better) |
| --- | --- | --- |
| Ayush v2 surrogate, ℝ³² | 0.921 | **0.297** |
| TRUE H0, ℝ³² | 0.629 | 1.118 |
| TRUE H0 + PCA-3 | 0.780 | 1.233 |
| **v2 surrogate + PCA-3** | **0.929** | 0.378 |

**Read — this partially refutes my own §7 recommendation from the research memo:**
- Swapping to true H0 persistence is a **bad idea**: MST death times are fine-scale
  nearest-neighbor distances, which iid noise and time-warps scramble (AUC 0.63,
  warp-intolerant). Ayush's top-k-largest-distances surrogate measures *global spread*,
  which is stable (AUC 0.92) and warp-tolerant.
- PCA-3 projection mildly improves the surrogate (0.921 → 0.929) — worth a training run,
  not a headline.
- **Refined diagnosis of the M4 robustness collapse:** the loss is *not* failing from
  noise. It is doing its job — anchoring the global trajectory shape to training
  geometry — and that is exactly wrong under camera-viewpoint shift, where the correct
  trajectory legitimately changes shape. Literature-consistent fix: anneal λ_PH to zero
  late in training, or apply it only on in-distribution-confident samples; not a
  different filtration.

## Exp C — POD eigen-motions (real chunks, 80/20 split)

| variance explained | 90% | 95% | 99% | 99.9% |
| --- | --- | --- | --- | --- |
| # POD modes needed | **13** | **23** | 64 | 119 |

Held-out reconstruction RMSE at equal basis size p:

| p | POD | DCT | random |
| --- | --- | --- | --- |
| 8 | **0.180** | 0.344 | 0.440 |
| 16 | **0.127** | 0.244 | 0.434 |
| 64 | **0.047** | 0.080 | 0.400 |

**Read:** real LIBERO-Spatial action chunks are strongly low-rank — 23 data-derived
eigen-motions carry 95% of the variance, and POD beats a DCT basis ~2× at small p. The
head's p=256 basis dimension is heavily over-provisioned (consistent with his own
p256→64 ablation losing little). The POD trunk (`pod_trunk.py`, 9.97M params, same
interface) gives an orthonormal, well-conditioned, *interpretable* basis — and the
eigen-motion plot (`results/expC_eigenmotions.png`) is a ready paper figure.

## GPU RESULT (2026-07-21, lab Blackwell RTX PRO 6000) — replan sweep on LIBERO-Long

Ran on Ayush's own trained **m3 30K Long checkpoint**, his `evaluate.py --replan`,
10 tasks × 20 episodes, no retraining. Rendering via osmesa (the box's EGL is broken by a
staged driver upgrade); policy inference on the Blackwell GPU.

| config | replan | LIBERO-Long mean SR |
| --- | --- | --- |
| m3_r5 (his default) | 5 | **60.00 %** |
| m3_r1 (the proposal) | 1 | **44.50 %** |

**Harness validated:** m3_r5 = 60.0 % reproduces his reported **58.5 %** (within seed noise),
so the offline pipeline is trustworthy.

**The replan-1 hypothesis is REFUTED — it is 15.5 points worse.** Per-task, replan-5 wins 8/10,
ties 1, loses 1:

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replan-1 | 15 | **45** | 70 | 95 | 25 | 75 | 35 | 30 | 5 | 50 |
| replan-5 | **45** | 30 | **90** | **100** | **60** | **80** | **65** | **55** | 5 | **70** |

**Interpretation.** Chunked execution is load-bearing for this head: re-predicting every step
lets the policy jump modes with no smoothing, which is exactly the reactivity-vs-consistency
trade-off BID (2408.17355) describes. It also matches Exp A above — systematic per-replan error
does not average out, and naive switching carries a boundary discontinuity every step.

**This makes the ensembling/pinning contribution the real one, not plain replan-1.** The open
question is now whether the loss comes from *observation freshness* (unlikely) or *unsmoothed
switching* (likely) — run replan-1 **+ fractional-τ ensembling / boundary pinning**
(`operator_chunk_exec.py`) against the two anchors above.

Still running in the same sweep: `flow_r5` (66.5 baseline), `m3_r2` (is the trend monotonic?),
`m4_r1`, `flow_r1`.

## What still needs the lab GPU (the actual SR numbers)

1. Replan-1 + pinning/ensembling eval on LIBERO-Long with existing checkpoints
   (`operator_chunk_exec.py` wraps the trained head as-is) — hours, no retraining.
2. POD-trunk training run at the standard recipe (`pod_trunk.PODHead` is interface-
   compatible with `DeepONetHeadV2` in `modeling_smolvla_deeponet_v2.py`).
3. λ_PH annealing run (config change) ± surrogate+PCA-3 (`ph_loss_stable._pca_project`
   composes with the existing `ph_surrogate_loss`).
