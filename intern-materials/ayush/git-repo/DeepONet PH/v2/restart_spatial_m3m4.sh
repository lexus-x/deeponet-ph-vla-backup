#!/usr/bin/env bash
# RESTART after disk-full crash (2026-06-15): flow_s0 checkpoint is intact, so we
# skip retraining flow. Remove the corrupt m3_s0 run, retrain M3 (deeponet) + M4
# (deeponet+PH) at 15K, then in-dist eval (flow/m3/m4) + plots + videos, and finally
# touch ../SPATIAL15K_DONE so the parked Object+Goal video re-eval auto-resumes.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

# Abort early if disk is dangerously low (avoid another truncated-checkpoint crash)
free_kb=$(df --output=avail / | tail -1)
if [ "$free_kb" -lt 10485760 ]; then echo "ABORT: <10GB free on /"; exit 1; fi

COMMON="--seed 0 --stage1_steps 1650 --stage2_steps 13350 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 --ckpt_every 20000 \
  --epoch_steps 200 --num_workers 8"
prune(){ local pd=$1; local latest; latest=$(cat "$pd/checkpoints/LATEST.txt")
  for c in "$pd"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done; }

SUITE=libero_spatial; DS=lerobot/libero_spatial_image; DIR=../Spatial
mkdir -p "$DIR/runs" "$DIR/logs" "$DIR/plots" "$DIR/videos" "$DIR/data"

echo "###### [RESTART] removing corrupt m3_s0 run ######"
rm -rf "$DIR/runs/m3_s0"

NAMES=(m3 m4); HEADS=(deeponet deeponet); VARS=(baseline ph)
for i in 0 1; do
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
echo "SPATIAL 15K (m3+m4) RETRAIN + EVAL DONE"
