# transfere — DeepONet (m3) + DeepONet+PH (m4) training on LIBERO

This package trains the **DeepONet operator action head** (`m3`) and **DeepONet + Persistent-Homology loss**
(`m4`) on **LIBERO-Spatial, -Object, and -Long**, using the **exact same recipe** as the flow-matching baseline
(30K steps, batch 48). It starts from the pretrained `lerobot/smolvla_base` (auto-downloaded) and swaps in the
DeepONet head — no checkpoints are shipped in this zip.

## Contents
```
code/
  train.py                          # trainer (flow | deeponet, baseline | ph)
  evaluate.py  evaluate_plus.py     # in-dist eval + LIBERO-Plus robustness eval
  modeling_smolvla_deeponet_v2.py   # DeepONet policy (the m3/m4 model)
  modeling_smolvla_ph.py            # flow/PH policy + dataset-feature adapter (imported by trainer)
  deeponet_head_v2.py               # the DeepONet operator head
  ph_loss.py                        # persistent-homology loss (used by m4)
  regression_head.py                # ablation head (not needed for m3/m4)
  libero_v_wrapper.py  libero_plus_wrapper.py   # env wrappers
  make_suite_plots.py               # per-suite comparison plots
  summarize_suite.py                # saves full + excl-lowest-task averages
  plot_10_and_9.py                  # the 10-task & 9-task annotated plots
  run_deeponet_m3m4.sh              # THE orchestration script (runs everything)
requirements.txt                    # pip freeze of the working env
flow_results/                       # flow baseline numbers for comparison (json/csv/png only)
SETUP_AND_RUN.md                    # this file
```

## The recipe (identical to the flow baseline)
| Setting | Value |
|---|---|
| Total steps | **30,000** = stage1 1,650 (head warm-up, backbone frozen) + stage2 28,350 (full fine-tune) |
| Batch size | **48** (both stages) |
| LR | head **1e-4**, backbone **1e-5** (500-step warmup) |
| EMA | **0.999** | 
| Checkpoint every | 10,000 steps (crash-safety; only LATEST kept) |
| Seed | 0 |
| DeepONet head | p=256, blocks=3, queries=8, fourier=16 |
| m4 PH loss | `--lambda_ph 0.02 --ph_k 8` |
| Eval | in-distribution, **20 episodes/task**, replan 5 |
| Datasets | `lerobot/libero_spatial_image`, `lerobot/libero_object_image`, `lerobot/libero_10_image` |

## Setup
```bash
# 1) Python 3.10–3.12 venv
python -m venv venv && source venv/bin/activate
pip install --upgrade pip

# 2) Dependencies (lerobot w/ smolvla, LIBERO, robosuite, torch, persim/ripser for PH, etc.)
pip install -r requirements.txt
#   If requirements.txt is too strict for this machine's CUDA, instead install the essentials:
#   pip install "lerobot[smolvla]" robosuite libero gymnasium imageio[ffmpeg] matplotlib numpy persim ripser
#   (LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO  — `pip install -e .` of that repo)

# 3) Hugging Face login (to pull smolvla_base + the LIBERO datasets)
hf auth login            # paste a READ token from YOUR account

# 4) headless rendering
export MUJOCO_GL=egl
```

## Run (detached — survives logout)
```bash
cd code
export MUJOCO_GL=egl
setsid nohup bash run_deeponet_m3m4.sh > run.out 2>&1 < /dev/null &
echo "started PID $!"
tail -f deeponet_results/PROGRESS.log     # watch progress
```

## Outputs
```
code/deeponet_results/<Suite>/
  runs/<m3|m4>_s0/checkpoints/LATEST/      # trained model
  runs/eval_<m3|m4>/success_rates.json     # per-task + average
  runs/eval_<m3|m4>/summary_full.json      # FULL 10-task average  (official number)
  runs/eval_<m3|m4>/summary_excl_task<N>.json  # 9-task average (lowest task removed)
  runs/eval_<m3|m4>/summary_both.csv
  plots/<suite>_all10tasks.png             # 10-task bar plot (avg annotated)
  plots/<suite>_excl_task<N>_9tasks.png    # 9-task bar plot  (avg annotated)
  logs/...
```

## Time / hardware
- ~30K steps ≈ 5–10 h per run on a single modern GPU (≈54 GB VRAM at batch 48 in stage2).
- 6 runs total → roughly **1.5–2.5 days** end-to-end. Reduce by lowering steps/batch if needed.

## Flow baseline (for comparison, from `flow_results/`)
Spatial flow 79.5% · Object flow 87.5% · Goal flow ~93.5% · Long flow (see file). Goal of m3/m4: **match
in-distribution and beat robustness** (DeepONet was ~2× more robust on LIBERO-Plus in the original study).
