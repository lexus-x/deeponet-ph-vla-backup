#!/usr/bin/env bash
# Clean LIBERO-Plus re-run for all 4 suites, fired AFTER the current re-eval finishes
# (RERUN_DONE), with NO concurrent diagnostic sims — isolates the Plus numbers from any
# interference by the earlier Object debugging runs. Same checkpoints, same protocol.
set -u
cd "/home/user/Desktop/Ayush PH test/ACT"
source ../venv/bin/activate
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=./act_results
PROG="$BASE/CLEAN_PLUS_PROGRESS.log"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }
NAMES=(Spatial Object Long Goal)
DATA=(lerobot/libero_spatial_image lerobot/libero_object_image lerobot/libero_10_image lerobot/libero_goal_image)
FLAG=(libero_spatial libero_object libero_10 libero_goal)
VARIANTS=(act act_deeponet act_deeponet_ph)
PLUS_MS=(300 300 600 300)   # Long gets the longer horizon, as in the main re-eval

note "waiting for current re-eval to finish (RERUN_DONE)…"
while [ ! -f "$BASE/RERUN_DONE" ]; do sleep 60; done
note "==== re-eval done — START clean LIBERO-Plus (all 4 suites, isolated) ===="
for k in 0 1 2 3; do
  nm=${NAMES[$k]}; ds=${DATA[$k]}; fl=${FLAG[$k]}; dir="$BASE/$nm"
  MS=""; for v in "${VARIANTS[@]}"; do MS="$MS --model ${v}=$dir/runs/${v}/checkpoints/LATEST"; done
  note "PLUS start $nm (max_steps=${PLUS_MS[$k]})"
  if python evaluate_plus_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_clean_plus" \
       --n_per_cat 12 --replan 5 --max_steps ${PLUS_MS[$k]} 2>&1 | tee "$dir/logs/clean_plus.log"; then
    note "PLUS done $nm"; else note "PLUS FAIL $nm"; fi
done
note "==== CLEAN PLUS DONE -> */runs/eval_clean_plus ===="; touch "$BASE/CLEAN_PLUS_DONE"
