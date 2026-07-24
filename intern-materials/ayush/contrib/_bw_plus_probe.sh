#!/bin/bash
# Check LIBERO-plus assets on blackwell; estimate POD ETA
set -u
ROOT="/home/user/Desktop/Ayush PH test"
echo "=== LIBERO-plus ==="
ls "$ROOT/third_party/LIBERO-plus" 2>/dev/null | head || echo missing
ls "$ROOT/third_party/" 2>/dev/null | head
echo "=== POD ==="
tail -8 "$ROOT/DeepONet PH/v2/pod_train_spatial/train.log"
ls "$ROOT/DeepONet PH/v2/pod_train_spatial/checkpoints" 2>/dev/null | head
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
# Object/Goal ckpts?
find "$ROOT" -maxdepth 5 -type d -name '30000' 2>/dev/null | head -40
