# Results — all tables

Model legend: **M1 = flow** (SmolVLA flow-matching baseline) · **M3 = DeepONet** operator head · **M4 = DeepONet + PH loss**.
Source files are noted per table.

---

## 1. Headline — LIBERO-Spatial, 5 seeds
`DeepONet PH/v2_results/data/summary.csv`

| Model | In-dist mean | In-dist std | Robust mean | Robust std | Latency (ms) | Head params (M) | Seeds |
|---|---|---|---|---|---|---|---|
| M1 flow | 79.4 | 1.4 | 17.9 | 5.2 | 148.1 | 99.9 | 5 |
| M3 DeepONet | 80.3 | 2.4 | **38.5** | 6.9 | **29.5** | **10.4** | 5 |
| M4 DeepONet+PH | **81.5** | 3.9 | 32.6 | 5.1 | 29.5 | 10.4 | 5 |

- In-dist: DeepONet ≥ flow. Robustness: DeepONet **>2×** flow. Params **9.6×** smaller, latency **~5×** faster.

## 2. Robustness by perturbation — LIBERO-Plus (Spatial)
`DeepONet PH/v2_results/data/robustness_per_category.csv`

| Perturbation | M1 flow | M3 DeepONet | M4 DeepONet+PH |
|---|---|---|---|
| Camera Viewpoints | 18.7 | **26.7** | 14.7 |
| Light Conditions | 22.7 | **61.3** | 54.7 |
| Sensor Noise | **25.3** | 21.3 | 17.3 |
| Background Textures | 24.0 | 42.7 | **44.0** |
| Objects Layout | 9.3 | **48.0** | 46.7 |
| Robot Initial States | 17.3 | **36.0** | 29.3 |
| Language Instructions | 8.0 | **33.3** | 21.3 |
| **Average** | 17.9 | **38.5** | 32.6 |

M3 wins **6/7** (flow wins only Sensor Noise). Biggest gains: Objects Layout (+38.7) and Language (+25.3).

## 3. Per-task — LIBERO-Spatial, 5 seeds
`DeepONet PH/v2_results/data/success_per_task.csv`

| Task | M1 flow | M3 DeepONet | M4 DeepONet+PH |
|---|---|---|---|
| task0 | 83.0 | 88.0 | **95.0** |
| task1 | 85.0 | **92.0** | **92.0** |
| task2 | **94.0** | 89.0 | 90.0 |
| task3 | **96.0** | 82.0 | 84.0 |
| task4 | 73.0 | 85.0 | **88.0** |
| task5 | 16.0 | 28.0 | **38.0** | ← the hard "stacked-bowl" task |
| task6 | **93.0** | 92.0 | 90.0 |
| task7 | 88.0 | 80.0 | 83.0 |
| task8 | 79.0 | **88.0** | 84.0 |
| task9 | **87.0** | 79.0 | 71.0 |

Note even on the hard task5, M3/M4 (28/38%) beat flow (16%).

## 4. Ablations — LIBERO-Spatial
`DeepONet PH/Ablation_Results/ablations.csv`

| Configuration | In-dist | Robust | Seeds |
|---|---|---|---|
| Full DeepONet-v2 (p256, 3 blk, Fourier) | 80.3 ± 2.4 | **38.5 ± 6.9** | 5 |
| (−) basis p 256→64 | 81.7 ± 0.5 | 32.7 ± 5.2 | 3 |
| (−) Fourier-τ (linear only) | 82.7 ± 1.4 | 39.0 ± 8.7 | 3 |
| (−) cross-attn blocks 3→1 | 78.2 ± 2.0 | 30.8 ± 4.5 | 3 |
| Regression head (no operator) | 83.0 ± 4.2 | 32.7 ± 0.9 | 3 |

## 5. Per-suite in-distribution — single seed, 15K steps
`DeepONet PH/{Spatial,Object,Goal}/runs/eval_indist/success_rates.json`

| Suite | M1 flow | M3 DeepONet | M4 DeepONet+PH |
|---|---|---|---|
| LIBERO-Spatial | 78.5 | **82.0** | 80.5 |
| LIBERO-Object | 84.5 | **94.0** | 87.0 |
| LIBERO-Goal | **93.5** | 90.0 | 89.0 |

### Per-task, single seed (for reference)
**Spatial flow:** t0 75 · t1 100 · t2 95 · t3 95 · t4 75 · **t5 5** · t6 95 · t7 85 · t8 70 · t9 90 → avg 78.5
**Object flow:** t0 75 · t1 75 · t2 100 · t3 90 · t4 100 · t5 80 · t6 80 · t7 75 · t8 100 · t9 70 → avg 84.5
**Goal flow:** t0 100 · t1 100 · t2 95 · t3 75 · t4 90 · t5 95 · t6 100 · t7 95 · t8 95 · t9 90 → avg 93.5

## 6. Live experiments (numbers land as they finish)
- **30K paper-repro campaign** — flow on Spatial/Object/Long. Status: `DeepONet PH/paper_repro/PROGRESS.log`,
  per-suite results in `paper_repro/{Spatial,Object,Long}/runs/eval_flow/`.
- **Goal object-layout test** — `DeepONet PH/goal_layouttest/off_{0,20,30}/success_rates.json`; aggregate via
  `DeepONet PH/v2/aggregate_goal_seedtest.py`.

---
*All numbers above are measured on this machine. Single-seed per-suite numbers carry ~±3% sampling noise (20
episodes/task). The cross-model parameter estimates in `architecture.md` use published model sizes (approximate).*
