#!/usr/bin/env bash
# 1) Goal videos (never ran). 2) Spatial 15K retrain: flow/M3-v2/M4-v2+PH, batch 48,
# in-dist eval + plots + videos. NO robustness (per user). Saves to ../Spatial/.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

# ---------- 1) Goal videos ----------
echo "###### Goal videos ######"
python make_videos_suite.py --suite libero_goal --dataset lerobot/libero_goal_image \
  --flow ../Goal/runs/flow_s0/checkpoints/LATEST \
  --m3 ../Goal/runs/m3_s0/checkpoints/LATEST \
  --m4 ../Goal/runs/m4_s0/checkpoints/LATEST \
  --out ../Goal/videos 2>&1 | tee ../Goal/logs/videos.log || echo "goal videos failed"
touch ../GOALVIDS_DONE

# ---------- 2) Spatial 15K retrain (no robustness) ----------
COMMON="--seed 0 --stage1_steps 1650 --stage2_steps 13350 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 --ckpt_every 20000 \
  --epoch_steps 200 --num_workers 8"
prune(){ local pd=$1; local latest; latest=$(cat "$pd/checkpoints/LATEST.txt")
  for c in "$pd"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done; }

SUITE=libero_spatial; DS=lerobot/libero_spatial_image; DIR=../Spatial
mkdir -p "$DIR/runs" "$DIR/logs" "$DIR/plots" "$DIR/videos" "$DIR/data"
NAMES=(flow m3 m4); HEADS=(flow deeponet deeponet); VARS=(baseline baseline ph)

rm -f ../SPATIAL15K_DONE
for i in 0 1 2; do
  nm=${NAMES[$i]}; hd=${HEADS[$i]}; vr=${VARS[$i]}
  echo "###### [SPATIAL15K] train $nm (head=$hd variant=$vr) 15K ######"
  python train.py --head $hd --variant $vr --dataset $DS --out $DIR/runs/${nm}_s0 $COMMON \
    2>&1 | tee $DIR/logs/train_${nm}.log
  prune $DIR/runs/${nm}_s0
done

echo "###### [SPATIAL15K] eval in-distribution ######"
python evaluate.py --suite $SUITE --dataset $DS \
  --model flow=flow=$DIR/runs/flow_s0/checkpoints/LATEST \
  --model m3=deeponet=$DIR/runs/m3_s0/checkpoints/LATEST \
  --model m4=deeponet=$DIR/runs/m4_s0/checkpoints/LATEST \
  --only indist --indist_episodes 20 --replan 5 --out $DIR/runs/eval_indist \
  2>&1 | tee $DIR/logs/eval_indist.log

echo "###### [SPATIAL15K] plots (in-dist) ######"
python make_suite_plots.py --suite_dir $DIR --suite $SUITE --label SPATIAL15K 2>&1 | tee $DIR/logs/plots.log || echo "plots failed"
echo "###### [SPATIAL15K] videos ######"
python make_videos_suite.py --suite $SUITE --dataset $DS \
  --flow $DIR/runs/flow_s0/checkpoints/LATEST \
  --m3 $DIR/runs/m3_s0/checkpoints/LATEST \
  --m4 $DIR/runs/m4_s0/checkpoints/LATEST \
  --out $DIR/videos 2>&1 | tee $DIR/logs/videos.log || echo "videos failed"

touch ../SPATIAL15K_DONE
echo "GOAL VIDEOS + SPATIAL 15K RETRAIN DONE"
