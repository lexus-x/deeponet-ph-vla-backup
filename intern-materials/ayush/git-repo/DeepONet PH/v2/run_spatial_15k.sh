#!/usr/bin/env bash
# Retrain LIBERO-Spatial at 15K steps (1650 stage1 + 13350 stage2), batch 48, 1 seed,
# 3 models (flow / DeepONet-v2 / DeepONet+PH-v2) -> stronger flow baseline (Spatial was
# undertrained at 7.5 epochs). WAITS for the robustness redo to free the GPU. Same recipe
# as Object/Goal so it's comparable. Saves into ../Spatial/. Detached + markers.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

until [ -f ../ROBUSTNESS_REDO_DONE ]; do sleep 300; done   # don't contend with the running redo

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
  echo "###### [SPATIAL15K] train $nm (head=$hd variant=$vr) 15K steps ######"
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

echo "###### [SPATIAL15K] eval robustness (LIBERO-Plus) ######"
python evaluate_plus.py --suite $SUITE --dataset $DS \
  --model flow=flow=$DIR/runs/flow_s0/checkpoints/LATEST \
  --model m3=deeponet=$DIR/runs/m3_s0/checkpoints/LATEST \
  --model m4=deeponet=$DIR/runs/m4_s0/checkpoints/LATEST \
  --n_per_cat 15 --replan 5 --max_steps 300 --out $DIR/runs/eval_plus \
  2>&1 | tee $DIR/logs/eval_plus.log

echo "###### [SPATIAL15K] plots ######"
python make_suite_plots.py --suite_dir $DIR --suite $SUITE --label SPATIAL15K 2>&1 | tee $DIR/logs/plots.log || echo "plots failed"
echo "###### [SPATIAL15K] videos ######"
python make_videos_suite.py --suite $SUITE --dataset $DS \
  --flow $DIR/runs/flow_s0/checkpoints/LATEST \
  --m3 $DIR/runs/m3_s0/checkpoints/LATEST \
  --m4 $DIR/runs/m4_s0/checkpoints/LATEST \
  --out $DIR/videos 2>&1 | tee $DIR/logs/videos.log || echo "videos failed"

touch ../SPATIAL15K_DONE
echo "SPATIAL 15K RETRAIN DONE"
