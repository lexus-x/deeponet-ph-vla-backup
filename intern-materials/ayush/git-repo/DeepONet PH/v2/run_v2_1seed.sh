#!/usr/bin/env bash
# DeepONet-v2 (cross-attention ~10.4M head), 1-seed check: M3-v2 + M4-v2 (seed 0).
# M1 (flow, 78.8%) is reused from the parent run. Reduced workers to share CPU
# with the robustness eval that may be running concurrently.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
mkdir -p logs runs

COMMON="--dataset lerobot/libero_spatial_image \
  --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 \
  --ckpt_every 3000 --epoch_steps 200 --num_workers 6 --deeponet_p 256"

rm -f V2_1SEED_DONE
echo "###### M3-v2 (DeepONet, cross-attn) seed0 ######"
python train.py --head deeponet --variant baseline --seed 0 --out runs/m3v2_s0 $COMMON \
  2>&1 | tee logs/train_m3v2_s0.log

echo "###### M4-v2 (DeepONet+PH lambda=0.02) seed0 ######"
python train.py --head deeponet --variant ph --lambda_ph 0.02 --seed 0 --out runs/m4v2_s0 $COMMON \
  2>&1 | tee logs/train_m4v2_s0.log

for d in runs/m3v2_s0 runs/m4v2_s0; do
  latest=$(cat "$d/checkpoints/LATEST.txt")
  for c in "$d"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done
done
touch V2_1SEED_DONE
echo "V2 1-SEED TRAINING DONE"
