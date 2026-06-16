#!/usr/bin/env bash
# ============================================================================
# Paper-reproduction campaign: 30K steps, batch 48, same SmolVLA training style.
# Goal: push FLOW to its paper values (Spatial 90 / Object 96 / Long >=71),
# then train m3 (DeepONet) + m4 (DeepONet+PH) for a budget-matched comparison.
#
# FLOW-FIRST ordering: all 3 flow runs + a flow-only eval happen first (~30 h),
# so the flow numbers land early; m3/m4 follow. Final 3-model eval at the end.
# Outputs in a SEPARATE folder (../paper_repro). Progress -> ../paper_repro/PROGRESS.log
# ============================================================================
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

BASE=../paper_repro
mkdir -p "$BASE"
PROG="$BASE/PROGRESS.log"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }

# 30K recipe (stage1 head-warmup 1650 + stage2 28350 = 30000); intermediate ckpts for crash-safety
COMMON="--seed 0 --stage1_steps 1650 --stage2_steps 28350 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 --ckpt_every 10000 \
  --epoch_steps 200 --num_workers 8"

prune(){ local pd=$1; local latest; latest=$(cat "$pd/checkpoints/LATEST.txt")
  for c in "$pd"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done; }

guard(){ local kb; kb=$(df --output=avail / | tail -1)
  if [ "$kb" -lt 10485760 ]; then note "ABORT: <10GB free on /"; exit 1; fi; }

# suite: NAME  DATASET  SUITE_FLAG
SNAME=(Spatial Object Long)
SDATA=(lerobot/libero_spatial_image lerobot/libero_object_image lerobot/libero_10_image)
SFLAG=(libero_spatial libero_object libero_10)

train_one(){ # $1 suite_idx  $2 nm  $3 head  $4 variant  $5 extra_args
  local i=$1 nm=$2 head=$3 var=$4 extra=$5
  local dir="$BASE/${SNAME[$i]}" ds="${SDATA[$i]}"
  mkdir -p "$dir/runs" "$dir/logs"
  guard
  note "TRAIN start  ${SNAME[$i]}/$nm  (head=$head variant=$var) 30K"
  python train.py --head $head --variant $var $extra --dataset "$ds" \
    --out "$dir/runs/${nm}_s0" $COMMON 2>&1 | tee "$dir/logs/train_${nm}.log"
  prune "$dir/runs/${nm}_s0"
  note "TRAIN done   ${SNAME[$i]}/$nm"
}

eval_suite(){ # $1 suite_idx  $2 out_subdir  $3... model specs
  local i=$1 out=$2; shift 2
  local dir="$BASE/${SNAME[$i]}"
  note "EVAL start   ${SNAME[$i]} -> $out"
  python evaluate.py --suite "${SFLAG[$i]}" --dataset "${SDATA[$i]}" "$@" \
    --only indist --indist_episodes 20 --replan 5 --out "$dir/runs/$out" \
    2>&1 | tee "$dir/logs/${out}.log"
  note "EVAL done    ${SNAME[$i]} -> $out (per-episode videos in runs/$out/episode_videos/)"
}

plot_suite(){ # $1 suite_idx  $2 eval_subdir  $3 label_suffix
  local i=$1 sub=$2 sfx=$3
  local dir="$BASE/${SNAME[$i]}"
  note "PLOTS start  ${SNAME[$i]} ($sub)"
  python make_suite_plots.py --suite_dir "$dir" --suite "${SFLAG[$i]}" \
    --label "${SNAME[$i]}${sfx}" --eval_subdir "$sub" \
    2>&1 | tee "$dir/logs/plots_${sub}.log" || note "plots failed ${SNAME[$i]} $sub"
  note "PLOTS done   ${SNAME[$i]} ($sub) -> accuracy + success_per_task in $dir/plots/"
}

note "==== CAMPAIGN START (30K, batch48, flow-first) ===="
rm -f "$BASE/FLOW_DONE" "$BASE/ALL_DONE"

# ---------- FLOW ONLY, per-suite: train -> eval -> plot each suite for EARLY incremental results ----------
for i in 0 1 2; do
  train_one $i flow flow baseline ""
  eval_suite $i eval_flow "--model flow=flow=$BASE/${SNAME[$i]}/runs/flow_s0/checkpoints/LATEST"
  plot_suite $i eval_flow _FLOW
  note "######## RESULT READY: ${SNAME[$i]} flow -> $BASE/${SNAME[$i]}/runs/eval_flow + $BASE/${SNAME[$i]}/plots ########"
done
touch "$BASE/FLOW_DONE"; touch "$BASE/ALL_DONE"
note "==== CAMPAIGN DONE (FLOW ONLY) ===="
