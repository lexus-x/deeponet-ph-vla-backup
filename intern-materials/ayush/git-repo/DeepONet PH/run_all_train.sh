#!/usr/bin/env bash
# Sequentially train M1 (flow baseline), M3 (DeepONet), M4 (DeepONet+PH) under an
# identical recipe on LIBERO-Spatial. ~1.4 h each, ~4.2 h total.
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

COMMON="--dataset lerobot/libero_spatial_image \
  --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 \
  --ckpt_every 3000 --epoch_steps 200 --num_workers 8"

echo "############ M1 flow baseline ############"
python train.py --head flow --variant baseline --out runs/m1_flow_baseline $COMMON \
  2>&1 | tee logs/train_m1.log

echo "############ M3 deeponet ############"
python train.py --head deeponet --variant baseline --out runs/m3_deeponet $COMMON \
  2>&1 | tee logs/train_m3.log

echo "############ M4 deeponet + PH (lambda=0.02) ############"
python train.py --head deeponet --variant ph --lambda_ph 0.02 --out runs/m4_deeponet_ph $COMMON \
  2>&1 | tee logs/train_m4.log

# prune intermediate checkpoints, keep only the final (LATEST) per model
for d in runs/m1_flow_baseline runs/m3_deeponet runs/m4_deeponet_ph; do
  latest=$(cat "$d/checkpoints/LATEST.txt")
  for c in "$d"/checkpoints/*/; do
    step=$(basename "$c")
    if [ "$step" != "$latest" ]; then rm -rf "$c"; fi
  done
  echo "pruned $d -> kept $latest"
done

touch TRAIN_ALL_COMPLETE
echo "ALL TRAINING DONE"
