# DeepONet-v2 (cross-attention) vs Flow-Matching

The v2 action head uses cross-attention pooling over the full VLM prefix. It matches flow in-distribution while being markedly more robust, at ~10x fewer head params and ~5x lower latency.

## Headline numbers (LIBERO-Spatial in-dist / LIBERO-Plus robustness)

| Model | In-dist acc | Robustness |
|---|---|---|
| M1 flow | 79.4% | 17.9% |
| M3 DeepONet-v2 | 80.3% | 38.5% |
| M4 DeepONet+PH-v2 | 81.5% | 32.6% |

## Plots (`plots/`)

- **1_accuracy.png** — in-distribution accuracy (success rate)
- **2_robustness.png** — total robustness across the 7 LIBERO-Plus perturbations
- **3_latency.png** — inference latency per action chunk (measured, batch 1)
- **4_parameters.png** — action-head parameter count (active params)
- **5_success_per_task.png** — success rate per LIBERO-Spatial task
- **6_robustness_by_perturbation.png** — robustness broken down by each LIBERO-Plus perturbation

## Video (`video/`)

**`v2_vs_flow__v2_WINS__pert_Lighting.mp4`** — side-by-side rollout on a LIBERO-Plus *Lighting* perturbation. M1 flow (left) fails under the lighting shift; DeepONet-v2 (right) completes the task (SUCCESS). This illustrates v2's out-of-distribution robustness advantage.

## Data (`data/`)
- `summary.csv` — per-model in-dist, robustness, latency, head params, #seeds
- `success_per_task.csv` — in-dist success rate per LIBERO-Spatial task
- `robustness_per_category.csv` — robustness per LIBERO-Plus perturbation type

_Numbers are seed means. Flow = 5 seeds; DeepONet decks as noted in summary.csv._
