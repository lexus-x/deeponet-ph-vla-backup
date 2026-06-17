#!/usr/bin/env bash
# Recovery: eval the ALREADY-TRAINED Spatial flow 30K checkpoint (campaign died on the
# eval arg-quoting bug, now fixed). Eval-only — no training. GPU is free.
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=../paper_repro
dir="$BASE/Spatial"
LOG="$BASE/PROGRESS.log"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

note "RECOVERY: Spatial flow 30K eval start (fixed --model)"
python evaluate.py --suite libero_spatial --dataset lerobot/libero_spatial_image \
  --model flow=flow=$dir/runs/flow_s0/checkpoints/LATEST \
  --only indist --indist_episodes 20 --replan 5 --out "$dir/runs/eval_flow" \
  2>&1 | tee "$dir/logs/eval_flow.log"
note "RECOVERY: Spatial flow 30K eval done"
python make_suite_plots.py --suite_dir "$dir" --suite libero_spatial \
  --label Spatial_FLOW --eval_subdir eval_flow 2>&1 | tee "$dir/logs/plots_eval_flow.log" || note "plots failed"
note "######## RESULT READY: Spatial flow 30K -> $dir/runs/eval_flow + $dir/plots ########"
touch "$BASE/SPATIAL_EVAL_DONE"
