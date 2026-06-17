#!/usr/bin/env bash
# ============================================================================
# Recovery run: train + eval FLOW at 30K for Object and Long (Spatial already done).
# Uses the FIXED --model arg. After each suite's eval: plots + saves results two ways
# (full all-task average AND average excluding the single lowest task). Resilient:
# if one suite fails, the other still runs. Outputs in ../paper_repro/{Object,Long}.
# ============================================================================
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

BASE=../paper_repro
PROG="$BASE/PROGRESS.log"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }
COMMON="--seed 0 --stage1_steps 1650 --stage2_steps 28350 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 --ckpt_every 10000 \
  --epoch_steps 200 --num_workers 8"
prune(){ local pd=$1; local latest; latest=$(cat "$pd/checkpoints/LATEST.txt")
  for c in "$pd"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done; }
guard(){ local kb; kb=$(df --output=avail / | tail -1)
  if [ "$kb" -lt 10485760 ]; then note "ABORT: <10GB free on /"; exit 1; fi; }

NAMES=(Object Long)
DATA=(lerobot/libero_object_image lerobot/libero_10_image)
FLAG=(libero_object libero_10)

note "==== OBJECT + LONG recovery run START (30K, batch48, flow) ===="
for k in 0 1; do
  nm=${NAMES[$k]}; ds=${DATA[$k]}; fl=${FLAG[$k]}
  dir="$BASE/$nm"; mkdir -p "$dir/runs" "$dir/logs"
  guard
  note "TRAIN start  $nm/flow 30K"
  if python train.py --head flow --variant baseline --dataset "$ds" \
        --out "$dir/runs/flow_s0" $COMMON 2>&1 | tee "$dir/logs/train_flow.log"; then
    prune "$dir/runs/flow_s0"
    note "TRAIN done   $nm/flow"
    guard
    note "EVAL start   $nm -> eval_flow"
    if python evaluate.py --suite "$fl" --dataset "$ds" \
          --model "flow=flow=$dir/runs/flow_s0/checkpoints/LATEST" \
          --only indist --indist_episodes 20 --replan 5 --out "$dir/runs/eval_flow" \
          2>&1 | tee "$dir/logs/eval_flow.log"; then
      note "EVAL done    $nm"
      python make_suite_plots.py --suite_dir "$dir" --suite "$fl" --label "${nm}_FLOW" \
        --eval_subdir eval_flow 2>&1 | tee "$dir/logs/plots_eval_flow.log" || note "plots failed $nm"
      python summarize_suite.py --suite_dir "$dir" --suite "$fl" --drop auto \
        2>&1 | tee "$dir/logs/summary_$nm.log" || note "summary failed $nm"
      note "######## RESULT READY: $nm flow 30K -> $dir/runs/eval_flow (full + excl-lowest) ########"
    else
      note "EVAL FAILED  $nm (kept checkpoint)"
    fi
  else
    note "TRAIN FAILED $nm — skipping to next suite"
  fi
done
note "==== OBJECT + LONG recovery run DONE ===="
touch "$BASE/OBJECT_LONG_DONE"
