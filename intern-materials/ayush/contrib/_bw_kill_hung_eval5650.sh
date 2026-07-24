#!/bin/bash
set -u
echo "=== diagnose hung train ==="
ps -o pid,stat,etime,pcpu,pmem,wchan:20,cmd -p 74783
ls -la "/home/user/Desktop/Ayush PH test/DeepONet PH/v2/pod_train_spatial/checkpoints/"
# any temp write?
find "/home/user/Desktop/Ayush PH test/DeepONet PH/v2/pod_train_spatial" -mmin -30 -type f 2>/dev/null | head -20
lsof -p 74783 2>/dev/null | grep -E 'safetensors|checkpoint|pod_train' | head -20
nvidia-smi
echo "=== kill hung train; eval ckpt 5650 ==="
# Kill train tree
pkill -f 'pod_train_spatial' || true
sleep 3
pgrep -af train.py || echo "train dead"
# Force mid eval with 5650
export FORCE_STEP=5650
ROOT="/home/user/Desktop/Ayush PH test/DeepONet PH/v2"
CONTRIB="/home/user/Desktop/Ayush PH test/contrib_postjul15"
VENV="/home/user/Desktop/Ayush PH test/venv/bin/python"
CKPT="$ROOT/pod_train_spatial/checkpoints/5650"
OUT="$ROOT/pod_eval_spatial_5650"
mkdir -p "$OUT"
echo 5650 > "$ROOT/pod_train_spatial/checkpoints/LATEST.txt"
export MUJOCO_GL=osmesa TOKENIZERS_PARALLELISM=false
export DEEPONET_HEAD=pod DEEPONET_P=32 DEEPONET_FOURIER=0 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8
export POD_CKPT="$ROOT/pod_assets/pod_p32_a32.pt"
export PYTHONPATH="$CONTRIB:$ROOT:${PYTHONPATH:-}"
NORM=policy_preprocessor_step_5_normalizer_processor.safetensors
echo "Evaluating CKPT=$CKPT" | tee "$OUT/run.log"
nohup env OFFLINE_STATS_SF="$CKPT/$NORM" "$VENV" "$CONTRIB/eval_exec_offline.py" \
  --suite libero_spatial --dataset lerobot/libero_spatial_image \
  --model "pod=deeponet=$CKPT" --exec none --replan 5 \
  --indist_episodes 10 --n_tasks 10 --out "$OUT" \
  >> "$OUT/run.log" 2>&1 &
echo POD_EVAL_PID=$!
sleep 8
tail -40 "$OUT/run.log"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
