#!/usr/bin/env bash
# Extend the MAIN comparison (M1 flow, M3-v2, M4-v2) to 5 seeds by adding seeds
# 3 & 4. Chains AFTER the seeds-0..2 v2 training finishes (waits for
# V2_ALLTRAIN_DONE) to avoid concurrent-training OOM. M1 flow trains in the parent
# dir; v2 models in this dir.
set -e
set -o pipefail
cd "$(dirname "$0")"            # .../DeepONet PH/v2
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

until [ -f V2_ALLTRAIN_DONE ]; do sleep 120; done

COMMON="--dataset lerobot/libero_spatial_image \
  --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 \
  --ckpt_every 3000 --epoch_steps 200 --num_workers 8"

rm -f SEED34_DONE
for S in 3 4; do
  echo "###### M3-v2 seed $S ######"
  python train.py --head deeponet --variant baseline --seed $S --out runs/m3v2_s$S $COMMON --deeponet_p 256 \
    2>&1 | tee logs/train_m3v2_s$S.log
  echo "###### M4-v2 seed $S ######"
  python train.py --head deeponet --variant ph --lambda_ph 0.02 --seed $S --out runs/m4v2_s$S $COMMON --deeponet_p 256 \
    2>&1 | tee logs/train_m4v2_s$S.log
  echo "###### M1 flow seed $S (parent dir) ######"
  ( cd .. && python train.py --head flow --variant baseline --seed $S --out runs/m1_flow_s$S $COMMON \
      2>&1 | tee logs/train_m1_s$S.log )
done

# prune intermediate checkpoints
for d in runs/m3v2_s3 runs/m4v2_s3 runs/m3v2_s4 runs/m4v2_s4 ../runs/m1_flow_s3 ../runs/m1_flow_s4; do
  latest=$(cat "$d/checkpoints/LATEST.txt"); for c in "$d"/checkpoints/*/; do
    [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done
done
touch SEED34_DONE
echo "SEEDS 3-4 TRAINED (M1 flow + M3-v2 + M4-v2)"
