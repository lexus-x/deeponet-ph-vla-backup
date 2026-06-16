#!/usr/bin/env bash
# Multi-seed training: M1 (flow), M3 (deeponet), M4 (deeponet+PH) x seeds {0,1,2}.
# Seed-outer ordering: a full 3-model comparison completes after seed 0 (~4.2h),
# with seeds 1-2 adding the error bars. ~12.6h total. ~1.4h per run.
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

COMMON="--dataset lerobot/libero_spatial_image \
  --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 \
  --ckpt_every 3000 --epoch_steps 200 --num_workers 8"

rm -f TRAIN_ALL_COMPLETE
for S in 0 1 2; do
  echo "################## SEED $S : M1 flow ##################"
  python train.py --head flow --variant baseline --seed $S \
    --out runs/m1_flow_s$S $COMMON 2>&1 | tee logs/train_m1_s$S.log

  echo "################## SEED $S : M3 deeponet ##################"
  python train.py --head deeponet --variant baseline --seed $S \
    --out runs/m3_deeponet_s$S $COMMON 2>&1 | tee logs/train_m3_s$S.log

  echo "################## SEED $S : M4 deeponet+PH (lambda=0.02) ##################"
  python train.py --head deeponet --variant ph --lambda_ph 0.02 --seed $S \
    --out runs/m4_deeponet_ph_s$S $COMMON 2>&1 | tee logs/train_m4_s$S.log

  # prune intermediate checkpoints for this seed (keep only final LATEST per run)
  for d in runs/m1_flow_s$S runs/m3_deeponet_s$S runs/m4_deeponet_ph_s$S; do
    latest=$(cat "$d/checkpoints/LATEST.txt")
    for c in "$d"/checkpoints/*/; do
      step=$(basename "$c")
      [ "$step" != "$latest" ] && rm -rf "$c" || true
    done
  done
  echo "=== seed $S complete (pruned) ==="
done

touch TRAIN_ALL_COMPLETE
echo "ALL 9 TRAINING RUNS DONE"
