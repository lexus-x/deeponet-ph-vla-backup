#!/usr/bin/env bash
# Boot overnight: kill sequential Plus, launch parallel Plus + POD30K + watcher.
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test/DeepONet PH/v2"

echo "=== killing sequential Plus ==="
# Kill wrapper + evaluate_plus for old campaign (not POD)
pkill -f '_bw_plus_multisuite.sh' 2>/dev/null || true
pkill -f 'plus_multisuite_campaign/m3_libero_object' 2>/dev/null || true
# More precise: kill evaluate_plus with n_per_cat 12 object only
pgrep -af evaluate_plus | grep 'libero_object' | grep 'n_per_cat 12' || true
pkill -f 'evaluate_plus.py --model m3=deeponet=.*Object' 2>/dev/null || true
sleep 2
# If still the old object m3 with n12:
for pid in $(pgrep -f 'evaluate_plus.py'); do
  cmd=$(ps -p "$pid" -o args=)
  if echo "$cmd" | grep -q 'n_per_cat 12'; then
    echo "killing n12 pid $pid"
    kill "$pid" 2>/dev/null || true
  fi
done
sleep 2

echo "=== remaining evaluate_plus ==="
pgrep -af evaluate_plus || echo none

echo "=== launch parallel Plus ==="
nohup bash /tmp/_bw_plus_parallel.sh > "$ROOT/plus_multisuite_campaign/parallel_outer.log" 2>&1 &
echo PLUS_ORCH=$!

echo "=== launch POD 30K ==="
bash /tmp/_bw_pod30k.sh

echo "=== launch watcher ==="
pkill -f '_bw_overnight_watcher.sh' 2>/dev/null || true
nohup bash /tmp/_bw_overnight_watcher.sh > "$ROOT/plus_multisuite_campaign/watcher.log" 2>&1 &
echo WATCHER=$!

sleep 12
echo "=== VERIFY ==="
pgrep -af 'evaluate_plus|_bw_plus_parallel|pod_train_spatial_30k|train.py|_bw_overnight_watcher' | grep -v grep | head -30
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
echo "=== logs ==="
tail -15 "$ROOT/plus_multisuite_campaign/parallel_outer.log" 2>/dev/null || true
tail -10 "$ROOT/pod_train_spatial_30k/train.log" 2>/dev/null || true
ls "$ROOT/plus_multisuite_campaign/" | head
