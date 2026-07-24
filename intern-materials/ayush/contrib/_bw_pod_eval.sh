#!/bin/bash
# On blackwell: when POD train finishes (or mid ckpt), run Spatial POD eval with osmesa.
set -u
ROOT="/home/user/Desktop/Ayush PH test/DeepONet PH/v2"
CONTRIB="/home/user/Desktop/Ayush PH test/contrib_postjul15"
VENV="/home/user/Desktop/Ayush PH test/venv/bin/python"
CKPT_DIR="$ROOT/pod_train_spatial/checkpoints"
# Prefer final if exists else latest numeric
if [[ -f "$CKPT_DIR/LATEST.txt" ]]; then
  STEP=$(cat "$CKPT_DIR/LATEST.txt")
  CKPT="$CKPT_DIR/$STEP"
else
  CKPT="$CKPT_DIR/3650"
fi
# Wait until train not running OR force mid
FORCE=${1:-}
if [[ -z "$FORCE" ]]; then
  if pgrep -f 'pod_train_spatial' >/dev/null; then
    echo "train still running; pass 'mid' to eval current LATEST anyway"
    exit 0
  fi
fi
OUT="$ROOT/pod_eval_spatial_${STEP:-mid}"
mkdir -p "$OUT"
export MUJOCO_GL=osmesa TOKENIZERS_PARALLELISM=false
export DEEPONET_HEAD=pod DEEPONET_P=32 DEEPONET_FOURIER=0 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8
export POD_CKPT="$ROOT/pod_assets/pod_p32_a32.pt"
export PYTHONPATH="$CONTRIB:$ROOT:${PYTHONPATH:-}"
NORM=policy_preprocessor_step_5_normalizer_processor.safetensors
echo "Evaluating CKPT=$CKPT -> $OUT"
nohup env OFFLINE_STATS_SF="$CKPT/$NORM" "$VENV" "$CONTRIB/eval_exec_offline.py" \
  --suite libero_spatial --dataset lerobot/libero_spatial_image \
  --model "pod=deeponet=$CKPT" --exec none --replan 5 \
  --indist_episodes 10 --n_tasks 10 --out "$OUT" \
  > "$OUT/run.log" 2>&1 &
echo POD_EVAL_PID=$! out=$OUT
