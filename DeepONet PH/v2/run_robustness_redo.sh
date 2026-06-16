#!/usr/bin/env bash
# Re-run robustness (LIBERO-Plus) for BOTH Object and Goal after the None-difficulty
# fix. Clears stale JSONs so it runs fresh. Same protocol as Spatial robustness
# (n_per_cat=15, replan=5, max_steps=300) for cross-suite comparability. Then
# regenerate plots for both + Goal videos (which never ran). Detached + markers.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

redo_robust(){
  local SUITE=$1 DS=$2 LABEL=$3 DIR=$4
  echo "###### [$LABEL] CLEAR stale robustness JSON + re-run ######"
  rm -f $DIR/runs/eval_plus/robustness_plus.json
  python evaluate_plus.py --suite $SUITE --dataset $DS \
    --model flow=flow=$DIR/runs/flow_s0/checkpoints/LATEST \
    --model m3=deeponet=$DIR/runs/m3_s0/checkpoints/LATEST \
    --model m4=deeponet=$DIR/runs/m4_s0/checkpoints/LATEST \
    --n_per_cat 15 --replan 5 --max_steps 300 --out $DIR/runs/eval_plus \
    2>&1 | tee $DIR/logs/eval_plus_redo.log
  echo "###### [$LABEL] refresh plots ######"
  python make_suite_plots.py --suite_dir $DIR --suite $SUITE --label $LABEL 2>&1 | tee $DIR/logs/plots_redo.log || echo "plots failed"
}

rm -f ../ROBUSTNESS_REDO_DONE
redo_robust libero_object lerobot/libero_object_image OBJECT ../Object
redo_robust libero_goal   lerobot/libero_goal_image   GOAL   ../Goal

echo "###### Goal videos (never ran originally) ######"
python make_videos_suite.py --suite libero_goal --dataset lerobot/libero_goal_image \
  --flow ../Goal/runs/flow_s0/checkpoints/LATEST \
  --m3 ../Goal/runs/m3_s0/checkpoints/LATEST \
  --m4 ../Goal/runs/m4_s0/checkpoints/LATEST \
  --out ../Goal/videos 2>&1 | tee ../Goal/logs/videos_redo.log || echo "goal videos failed"

touch ../GOAL_DONE ../OBJGOAL_DONE ../ROBUSTNESS_REDO_DONE
echo "ROBUSTNESS REDO (Object+Goal) + Goal videos DONE"
