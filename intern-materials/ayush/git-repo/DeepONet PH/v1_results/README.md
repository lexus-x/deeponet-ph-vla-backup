# DeepONet-v1 (mean-pool, NO cross-attention) vs Flow-Matching

The v1 action head pools the VLM prefix to a single vector (no cross-attention). It is tiny/fast (2.31M, 23ms) but trades a large chunk of accuracy and does not beat flow on robustness.

## Headline numbers (LIBERO-Spatial in-dist / LIBERO-Plus robustness)

| Model | In-dist acc | Robustness |
|---|---|---|
| M1 flow | 79.4% | 17.9% |
| M3 DeepONet-v1 | 53.0% | 15.6% |
| M4 DeepONet+PH-v1 | 54.2% | 17.1% |

## Plots (`plots/`)

- **1_accuracy.png** — in-distribution accuracy (success rate)
- **2_robustness.png** — total robustness across the 7 LIBERO-Plus perturbations
- **3_latency.png** — inference latency per action chunk (measured, batch 1)
- **4_parameters.png** — action-head parameter count (active params)
- **5_success_per_task.png** — success rate per LIBERO-Spatial task
- **6_robustness_by_perturbation.png** — robustness broken down by each LIBERO-Plus perturbation

## Video (`video/`)

**`flow_vs_v1__flow_WINS__indist_task08.mp4`** — side-by-side rollout on LIBERO-Spatial task 8. M1 flow (left) completes the pick-and-place (SUCCESS); DeepONet-v1 (right) fails. This is the case the mean-pool bottleneck motivated fixing in v2.

## Data (`data/`)
- `summary.csv` — per-model in-dist, robustness, latency, head params, #seeds
- `success_per_task.csv` — in-dist success rate per LIBERO-Spatial task
- `robustness_per_category.csv` — robustness per LIBERO-Plus perturbation type

_Numbers are seed means. Flow = 5 seeds; DeepONet decks as noted in summary.csv._
