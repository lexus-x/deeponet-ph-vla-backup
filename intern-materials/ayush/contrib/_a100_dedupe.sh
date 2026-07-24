#!/bin/bash
# Kill duplicate long confirm (keep oldest)
set -u
pids=$(pgrep -f '_a100_long_confirm.sh' | sort -n)
echo "bash pids: $pids"
# Keep first bash, kill later ones and their children
first=$(echo "$pids" | head -1)
for p in $pids; do
  if [[ "$p" != "$first" ]]; then
    echo "Killing duplicate bash $p and children"
    pkill -P "$p" 2>/dev/null || true
    kill "$p" 2>/dev/null || true
  fi
done
sleep 2
ps -ef | grep eval_exec | grep -v grep
# show progress
grep 'LIBERO-10' ~/deeponet_campaign/long_confirm_results/run.log | tail -20
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
