#!/usr/bin/env bash
# Re-run Object + Goal IN-DIST eval with the new per-episode video saving, and
# delete their old (single-rollout) videos. WAITS for the Spatial retrain to finish
# (SPATIAL15K_DONE) so it doesn't contend for the GPU. Per-episode videos land in
# {suite}/videos/{model}/ named task{N}_{taskname}_ep{E}_{PASS|FAIL}.mp4.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

until [ -f ../SPATIAL15K_DONE ]; do sleep 300; done   # don't contend with the Spatial retrain

reeval(){
  local SUITE=$1 DS=$2 LABEL=$3 DIR=$4
  echo "###### [$LABEL] delete old videos + clear in-dist JSON (force fresh eval) ######"
  rm -rf $DIR/videos/*
  rm -f $DIR/runs/eval_indist/success_rates.json
  rm -rf $DIR/runs/eval_indist/episode_videos
  echo "###### [$LABEL] re-run in-dist eval (saves per-episode videos) ######"
  python evaluate.py --suite $SUITE --dataset $DS \
    --model flow=flow=$DIR/runs/flow_s0/checkpoints/LATEST \
    --model m3=deeponet=$DIR/runs/m3_s0/checkpoints/LATEST \
    --model m4=deeponet=$DIR/runs/m4_s0/checkpoints/LATEST \
    --only indist --indist_episodes 20 --replan 5 --out $DIR/runs/eval_indist \
    2>&1 | tee $DIR/logs/eval_indist_revid.log
  echo "###### [$LABEL] move per-episode videos -> $DIR/videos/ ######"
  mkdir -p $DIR/videos
  mv $DIR/runs/eval_indist/episode_videos/* $DIR/videos/ 2>/dev/null || true
  rmdir $DIR/runs/eval_indist/episode_videos 2>/dev/null || true
  echo "###### [$LABEL] refresh in-dist plots ######"
  python make_suite_plots.py --suite_dir $DIR --suite $SUITE --label $LABEL 2>&1 | tee $DIR/logs/plots_revid.log || echo "plots failed"
  echo "###### [$LABEL] video count: $(ls $DIR/videos/*/*.mp4 2>/dev/null | wc -l) ######"
}

rm -f ../OBJGOAL_VIDS_DONE
reeval libero_object lerobot/libero_object_image OBJECT ../Object
reeval libero_goal   lerobot/libero_goal_image   GOAL   ../Goal
touch ../OBJGOAL_VIDS_DONE
echo "OBJECT+GOAL PER-EPISODE VIDEO RE-EVAL DONE"
