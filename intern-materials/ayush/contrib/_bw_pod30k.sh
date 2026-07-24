#!/usr/bin/env bash
# Matched-budget POD Spatial train (~30K recipe).
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test"
V2="$ROOT/DeepONet PH/v2"
VENV="$ROOT/venv/bin/python"
OUT="$V2/pod_train_spatial_30k"
CONTRIB="$ROOT/contrib_postjul15"
mkdir -p "$OUT"
cd "$V2"

export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export DEEPONET_HEAD=pod DEEPONET_P=32 DEEPONET_FOURIER=0 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8
export POD_CKPT="$V2/pod_assets/pod_p32_a32.pt"
export PYTHONPATH="$CONTRIB:$V2:${PYTHONPATH:-}"

# Avoid duplicate
if pgrep -f 'pod_train_spatial_30k' >/dev/null; then
  echo "POD 30K already running"
  pgrep -af 'pod_train_spatial_30k' | head -3
  exit 0
fi

echo "POD30K_START $(date)" | tee "$OUT/train.log"
nohup "$VENV" train.py \
  --head deeponet --variant baseline --deeponet_head pod \
  --deeponet_p 32 --deeponet_fourier 0 \
  --out "$OUT" \
  --dataset lerobot/libero_spatial_image --seed 0 \
  --stage1_steps 6000 --stage2_steps 24000 \
  >> "$OUT/train.log" 2>&1 &
echo POD30K_PID=$!
sleep 3
tail -20 "$OUT/train.log" || true
