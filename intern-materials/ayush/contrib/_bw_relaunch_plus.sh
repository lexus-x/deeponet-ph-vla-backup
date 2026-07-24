#!/usr/bin/env bash
# Relaunch Plus only (POD/watcher already up).
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test/DeepONet PH/v2"
pkill -f '_bw_plus_parallel.sh' 2>/dev/null || true
# don't kill n8 evaluate_plus if any good ones; kill nothing if none
sleep 1
nohup bash /tmp/_bw_plus_parallel.sh >> "$ROOT/plus_multisuite_campaign/parallel_outer.log" 2>&1 &
echo PLUS_ORCH=$!
sleep 10
pgrep -af 'evaluate_plus|_bw_plus_parallel' | grep -v grep | head -20
tail -30 "$ROOT/plus_multisuite_campaign/parallel_run.log"
ls -la "$ROOT/plus_multisuite_campaign/" | head -30
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
pgrep -af 'pod_train_spatial_30k|train.py' | grep -v grep | head -5
pgrep -af overnight_watcher | head -3
