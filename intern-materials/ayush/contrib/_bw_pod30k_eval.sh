#!/usr/bin/env bash
# Method track A: POD-30K Spatial in-dist vs Fourier M3 anchor 77%.
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test"
V2="$ROOT/DeepONet PH/v2"
CONTRIB="$ROOT/contrib_postjul15"
VENV="$ROOT/venv/bin/python"
CKPT="$V2/pod_train_spatial_30k/checkpoints/30000"
OUT="$V2/pod30k_eval_spatial"
mkdir -p "$OUT"
export MUJOCO_GL=osmesa TOKENIZERS_PARALLELISM=false
export DEEPONET_HEAD=pod DEEPONET_P=32 DEEPONET_FOURIER=0 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8
export POD_CKPT="$V2/pod_assets/pod_p32_a32.pt"
export PYTHONPATH="$CONTRIB:$V2:${PYTHONPATH:-}"
NORM=policy_preprocessor_step_5_normalizer_processor.safetensors
echo "POD30K_EVAL_START $(date)" | tee "$OUT/run.log"
nohup env OFFLINE_STATS_SF="$CKPT/$NORM" "$VENV" "$CONTRIB/eval_exec_offline.py" \
  --suite libero_spatial --dataset lerobot/libero_spatial_image \
  --model "pod=deeponet=$CKPT" --exec none --replan 5 \
  --indist_episodes 10 --n_tasks 10 --out "$OUT" \
  >> "$OUT/run.log" 2>&1 &
echo POD30K_EVAL_PID=$!
