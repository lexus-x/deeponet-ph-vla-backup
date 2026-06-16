#!/usr/bin/env bash
# Generalization run: LIBERO-Object + LIBERO-Goal, 1 seed, 15K steps (1650 stage1 +
# 13350 stage2), 3 models each (flow / DeepONet-v2 / DeepONet+PH-v2). Per suite:
# train 3 -> eval in-dist -> eval robustness (LIBERO-Plus) -> plots -> videos.
# Saves into ../Object/ and ../Goal/. Detached + marker files + autonomous.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

COMMON="--seed 0 --stage1_steps 1650 --stage2_steps 13350 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 --ckpt_every 20000 \
  --epoch_steps 200 --num_workers 8"

prune(){ local pd=$1; local latest; latest=$(cat "$pd/checkpoints/LATEST.txt")
  for c in "$pd"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done; }

# model table: name | head | variant
NAMES=(flow m3 m4); HEADS=(flow deeponet deeponet); VARS=(baseline baseline ph)

run_suite(){
  local SUITE=$1 DS=$2 LABEL=$3 DIR=$4      # e.g. libero_object  lerobot/libero_object_image  OBJECT  ../Object
  mkdir -p "$DIR/runs" "$DIR/logs" "$DIR/plots" "$DIR/videos" "$DIR/data"
  echo "###################### SUITE $LABEL ($SUITE) ######################"
  for i in 0 1 2; do
    local nm=${NAMES[$i]} hd=${HEADS[$i]} vr=${VARS[$i]}
    echo "###### [$LABEL] train $nm (head=$hd variant=$vr) 15K steps ######"
    python train.py --head $hd --variant $vr --dataset $DS --out $DIR/runs/${nm}_s0 $COMMON \
      2>&1 | tee $DIR/logs/train_${nm}.log
    prune $DIR/runs/${nm}_s0
  done
  echo "###### [$LABEL] eval in-distribution ######"
  python evaluate.py --suite $SUITE --dataset $DS \
    --model flow=flow=$DIR/runs/flow_s0/checkpoints/LATEST \
    --model m3=deeponet=$DIR/runs/m3_s0/checkpoints/LATEST \
    --model m4=deeponet=$DIR/runs/m4_s0/checkpoints/LATEST \
    --only indist --indist_episodes 20 --replan 5 --out $DIR/runs/eval_indist \
    2>&1 | tee $DIR/logs/eval_indist.log
  echo "###### [$LABEL] eval robustness (LIBERO-Plus) ######"
  python evaluate_plus.py --suite $SUITE --dataset $DS \
    --model flow=flow=$DIR/runs/flow_s0/checkpoints/LATEST \
    --model m3=deeponet=$DIR/runs/m3_s0/checkpoints/LATEST \
    --model m4=deeponet=$DIR/runs/m4_s0/checkpoints/LATEST \
    --n_per_cat 15 --replan 5 --out $DIR/runs/eval_plus \
    2>&1 | tee $DIR/logs/eval_plus.log
  echo "###### [$LABEL] plots ######"
  python make_suite_plots.py --suite_dir $DIR --suite $SUITE --label $LABEL 2>&1 | tee $DIR/logs/plots.log || echo "plots failed"
  echo "###### [$LABEL] videos ######"
  python make_videos_suite.py --suite $SUITE --dataset $DS \
    --flow $DIR/runs/flow_s0/checkpoints/LATEST \
    --m3 $DIR/runs/m3_s0/checkpoints/LATEST \
    --m4 $DIR/runs/m4_s0/checkpoints/LATEST \
    --out $DIR/videos 2>&1 | tee $DIR/logs/videos.log || echo "videos failed"
  echo "###### [$LABEL] DONE ######"
}

rm -f ../OBJGOAL_DONE ../OBJECT_DONE ../GOAL_DONE
run_suite libero_object lerobot/libero_object_image OBJECT ../Object
touch ../OBJECT_DONE
run_suite libero_goal   lerobot/libero_goal_image   GOAL   ../Goal
touch ../GOAL_DONE
touch ../OBJGOAL_DONE
echo "ALL OBJECT+GOAL DONE"
