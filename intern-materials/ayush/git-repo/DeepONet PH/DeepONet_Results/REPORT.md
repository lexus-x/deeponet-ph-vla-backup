# DeepONet vs Flow-matching — Final Report

## Headline (LIBERO-Spatial, replan-5, mean ± std over seeds)

| Model | In-dist acc (%) | Robustness (%) | Head params | Latency |
|---|---|---|---|---|
| M1 flow | 79.4 ± 1.4 (n=5) | 17.9 ± 5.2 (n=5) | 99.9M | 148ms |
| M3 DeepONet v1 | 53.0 ± 3.2 (n=3) | 15.6 ± 4.7 (n=3) | 2.3M | 23ms |
| M4 DON+PH v1 | 54.2 ± 3.4 (n=3) | 17.1 ± 2.1 (n=3) | 2.3M | 23ms |
| M3 DeepONet v2 | 80.3 ± 2.4 (n=5) | 38.5 ± 6.9 (n=5) | ~10.4M | ~25ms |
| M4 DON+PH v2 | 81.5 ± 3.9 (n=5) | 32.6 ± 5.1 (n=5) | ~10.4M | ~25ms |

## Paired significance (positive mean_diff favours first model)

| comparison | metric | n | Δ (pp) | t p-value | Wilcoxon p |
|---|---|---|---|---|---|
| don_v2_vs_flow | in-dist | 50 | +0.9 | 0.605 | 0.479 |
| don_v2_vs_flow | robust | 35 | +20.6 | 0.000 | 0.000 |
| donph_v2_vs_flow | in-dist | 50 | +2.1 | 0.355 | 0.381 |
| donph_v2_vs_flow | robust | 35 | +14.7 | 0.000 | 0.000 |
| don_v2_vs_don_v1 | in-dist | 30 | +28.5 | 0.000 | 0.000 |
| don_v2_vs_don_v1 | robust | 21 | +19.4 | 0.001 | 0.002 |
| donph_v2_vs_don_v2 | in-dist | 50 | +1.2 | 0.507 | 0.472 |
| donph_v2_vs_don_v2 | robust | 35 | -5.9 | 0.032 | 0.056 |
| don_v1_vs_flow | in-dist | 30 | -25.8 | 0.000 | 0.000 |
| don_v1_vs_flow | robust | 21 | -4.8 | 0.416 | 0.313 |

## Plots
- accuracy_robustness.png
- per_task_v1_v2.png
- robustness_per_category.png

_Significance via paired t-test + Wilcoxon over matched (task,seed) / (category,seed) units._