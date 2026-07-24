#!/usr/bin/env bash
# Clean re-evaluation of all 4 LIBERO suites from the already-trained checkpoints.
#   - waits for the training campaign to finish (ALL_DONE) so it never contends for GPU
#   - in-distribution: 3 seeds (the bug-fixed evaluate_act.py)
#   - LIBERO-Plus robustness
#   - writes to SEPARATE eval_rerun_* dirs (originals untouched)
#   - IDENTICAL protocol/args for every model — no per-model tuning, no cherry-picking
set -u
cd "/home/user/Desktop/Ayush PH test/ACT"
source ../venv/bin/activate
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1

BASE=./act_results
PROG="$BASE/RERUN_EVAL_PROGRESS.log"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }

NAMES=(Spatial Object Long Goal)
DATA=(lerobot/libero_spatial_image lerobot/libero_object_image lerobot/libero_10_image lerobot/libero_goal_image)
FLAG=(libero_spatial libero_object libero_10 libero_goal)
VARIANTS=(act act_deeponet act_deeponet_ph)
INDIST_BASE="--indist_episodes 10 --test_seeds 3 --replan 5 --only indist"
PLUS_BASE="--n_per_cat 12 --replan 5"
# Eval horizon must fit the task length. Long (k=2) has ~268-step demos, so the
# default 300 truncated it; give Long room (SAME budget for all 3 models). The
# short suites (~120-150 steps) already have ample headroom -> keep original budget.
INDIST_MS=(520 520 700 520)
PLUS_MS=(300 300 600 300)

note "waiting for training campaign to finish (ALL_DONE)…"
while [ ! -f "$BASE/ALL_DONE" ]; do sleep 60; done
note "==== campaign done — START clean re-eval (3-seed in-dist + LIBERO-Plus, all suites) ===="

for k in 0 1 2 3; do
  nm=${NAMES[$k]}; ds=${DATA[$k]}; fl=${FLAG[$k]}; dir="$BASE/$nm"
  MS=""; for v in "${VARIANTS[@]}"; do MS="$MS --model ${v}=$dir/runs/${v}/checkpoints/LATEST"; done
  INDIST="$INDIST_BASE --max_steps ${INDIST_MS[$k]}"
  PLUS="$PLUS_BASE --max_steps ${PLUS_MS[$k]}"

  note "INDIST start $nm (3-seed, max_steps=${INDIST_MS[$k]})"
  if python evaluate_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_rerun_indist" $INDIST \
       2>&1 | tee "$dir/logs/rerun_indist.log"; then note "INDIST done $nm"; else note "INDIST FAIL $nm"; fi

  note "PLUS start $nm"
  if python evaluate_plus_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_rerun_plus" $PLUS \
       2>&1 | tee "$dir/logs/rerun_plus.log"; then note "PLUS done $nm"; else note "PLUS FAIL $nm"; fi
done
note "==== RE-EVAL DONE -> */runs/eval_rerun_* ===="
touch "$BASE/RERUN_DONE"
