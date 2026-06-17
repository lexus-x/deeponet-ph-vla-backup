# transfere — DeepONet (m3) + DeepONet+PH (m4) training on LIBERO Spatial / Object / Long

Trains the **DeepONet operator action head** (`m3 = deeponet/baseline`) and
**DeepONet + Persistent-Homology loss** (`m4 = deeponet/ph`) on **LIBERO-Spatial,
-Object, and -Long** using the **exact same recipe as the flow-matching baseline**
(30,000 steps, batch 48, two-stage fine-tune). It starts from the pretrained
`lerobot/smolvla_base` (auto-downloaded) and swaps in the DeepONet head.

**No model checkpoints, no datasets, and no LIBERO repo are bundled** — the base
model + LIBERO datasets auto-download from Hugging Face; LIBERO itself is cloned
on the new machine (see Setup).

## Contents
```
code/
  train.py                          # one trainer for flow | deeponet, baseline | ph
  evaluate.py                       # in-distribution LIBERO eval (20 ep/task, replan 5)
  evaluate_plus.py                  # OPTIONAL LIBERO-Plus robustness eval
  modeling_smolvla_deeponet_v2.py   # the m3/m4 policy (DeepONet head on SmolVLA)
  modeling_smolvla_ph.py            # flow/PH policy + dataset-feature adapter (imported by trainer)
  deeponet_head_v2.py               # the DeepONet operator head
  ph_loss.py                        # persistent-homology surrogate loss (pure torch; m4 only)
  regression_head.py                # ablation head (not used by m3/m4; harmless)
  libero_v_wrapper.py               # in-dist env wrapper (uses lerobot LiberoEnv)
  libero_plus_wrapper.py            # robustness env wrapper (LIBERO-Plus; optional)
  summarize_suite.py                # saves full + excl-lowest-task averages
  plot_10_and_9.py                  # 10-task & 9-task annotated bar plots
  make_suite_plots.py               # per-suite comparison plots
  run_deeponet_m3m4.sh              # THE orchestration script — runs all 6 train+eval jobs
requirements.txt                    # exact pip freeze of the working env (Python 3.12)
flow_results/                       # flow baseline numbers for comparison (json/csv/png; NO videos)
  Spatial/  Object/                 # (Long flow was still training at packaging time)
SETUP_AND_RUN.md                    # this file
```

## The recipe (identical to the flow baseline)
| Setting | Value |
|---|---|
| Total steps | **30,000** = stage1 **1,650** (head warm-up, backbone frozen) + stage2 **28,350** (full fine-tune) |
| Batch size | **48** (both stages) |
| Learning rate | head **1e-4**, backbone **1e-5** (500-step linear warmup on backbone) |
| Optimizer | AdamW, betas (0.9, 0.95), weight_decay 1e-6, grad-clip 10.0 |
| Precision | bf16 autocast + gradient checkpointing in stage 2 |
| EMA | **0.999** (checkpoints save EMA weights) |
| Checkpoint every | 10,000 steps (crash-safety; only LATEST is kept) |
| epoch_steps / num_workers | 200 / 8 |
| Seed | 0 |
| DeepONet head | p=256, blocks=3, queries=8, fourier=16 |
| m4 PH loss | `--lambda_ph 0.02 --ph_k 8` |
| Base model | `lerobot/smolvla_base` (auto-download) |
| Datasets | `lerobot/libero_spatial_image`, `lerobot/libero_object_image`, `lerobot/libero_10_image` |
| Eval | in-distribution, **20 episodes/task**, replan 5 |

## Setup
```bash
# 1) Python 3.10–3.12 virtual env
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip

# 2) Python deps. Try the pinned set first:
pip install -r requirements.txt
#    If a pin conflicts with this machine's CUDA/driver, install the essentials instead
#    (let pip resolve a torch build that matches the local CUDA):
#    pip install "lerobot[smolvla]" robosuite==1.4.0 gymnasium imageio[ffmpeg] \
#                matplotlib numpy mujoco torch torchvision
#    NOTE: persim/ripser/gudhi are NOT needed — ph_loss.py is a pure-torch surrogate.

# 3) Install LIBERO (NOT in the zip — provides the simulator tasks/assets)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
#    (LIBERO-Plus is only needed for the OPTIONAL evaluate_plus.py robustness eval.)

# 4) Hugging Face login — pulls smolvla_base + the LIBERO datasets
hf auth login          # paste a READ token

# 5) Headless GPU rendering
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
The script runs **6 training jobs** (m3 + m4 × Spatial/Object/Long), then evaluates
each in-distribution, makes 10-task & 9-task plots, and writes both-way summaries.
It is resilient (one failure doesn't stop the rest) and crash-safe (intermediate ckpts).

## Outputs
```
code/deeponet_results/<Suite>/
  runs/<m3|m4>_s0/checkpoints/LATEST/        # trained model (EMA weights)
  runs/eval_<m3|m4>/success_rates.json       # per-task + average
  runs/eval_<m3|m4>/summary_full.json        # FULL 10-task average  (OFFICIAL number)
  runs/eval_<m3|m4>/summary_excl_task<N>.json# 9-task average (lowest task removed)
  runs/eval_<m3|m4>/summary_both.csv
  plots/<suite>_all10tasks.png               # 10-task bar plot
  plots/<suite>_excl_task<N>_9tasks.png      # 9-task bar plot
  logs/...   PROGRESS.log
```

## Time / hardware
- ~30K steps ≈ **8–10 h per run** on a single modern GPU (~54 GB VRAM at batch 48 in stage 2).
- 6 runs total → roughly **2–2.5 days** end-to-end. If VRAM is tighter, lower `--stage2_batch`
  (keep `--stage1_batch` equal) — but for an apples-to-apples comparison with flow, keep 48.

## Flow baseline (for comparison — from `flow_results/`)
| Suite | Flow, full 10-task (official) | Flow, excl-lowest 9-task |
|---|---|---|
| Spatial | **79.5%** | 87.78% (excl task5) |
| Object  | **87.5%** | 90.56% (excl task9) |
| Long    | _still training at packaging time — read its summary on the source machine_ |

Goal for m3/m4: **match in-distribution accuracy and beat robustness** (DeepONet was
~2× more robust on LIBERO-Plus in the original study).
