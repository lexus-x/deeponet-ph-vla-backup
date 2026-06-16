#!/usr/bin/env bash
# Train the remaining v2 seeds (M3-v2 + M4-v2, seeds 1 & 2) to complete the
# 3-seed study. WAITS for the seed-0 run to finish first (avoids GPU OOM from two
# concurrent trainings). ~5.6h. Full workers since the v1 eval is already done.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

until [ -f V2_1SEED_DONE ]; do sleep 120; done   # wait out seed-0 training

COMMON="--dataset lerobot/libero_spatial_image \
  --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 \
  --ckpt_every 3000 --epoch_steps 200 --num_workers 8 --deeponet_p 256"

rm -f V2_ALLTRAIN_DONE
for S in 1 2; do
  echo "###### M3-v2 seed $S ######"
  python train.py --head deeponet --variant baseline --seed $S --out runs/m3v2_s$S $COMMON \
    2>&1 | tee logs/train_m3v2_s$S.log
  echo "###### M4-v2 seed $S ######"
  python train.py --head deeponet --variant ph --lambda_ph 0.02 --seed $S --out runs/m4v2_s$S $COMMON \
    2>&1 | tee logs/train_m4v2_s$S.log
done

for d in runs/m3v2_s1 runs/m4v2_s1 runs/m3v2_s2 runs/m4v2_s2; do
  latest=$(cat "$d/checkpoints/LATEST.txt")
  for c in "$d"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done
done
touch V2_ALLTRAIN_DONE
echo "V2 ALL SEEDS TRAINED"
