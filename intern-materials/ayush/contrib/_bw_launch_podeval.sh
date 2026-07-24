#!/bin/bash
set -u
pkill -f 'while pgrep -f' 2>/dev/null || true
bash /tmp/_bw_pod_eval.sh mid
sleep 12
echo "=== procs ==="
pgrep -af eval_exec || echo none
echo "=== dirs ==="
ls -d /home/user/Desktop/Ayush\ PH\ test/DeepONet\ PH/v2/pod_eval* 2>/dev/null || echo no_eval_dirs
echo "=== logs ==="
for d in /home/user/Desktop/Ayush\ PH\ test/DeepONet\ PH/v2/pod_eval*; do
  echo "== $d"
  tail -50 "$d/run.log" 2>/dev/null || true
done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
cat /home/user/Desktop/Ayush\ PH\ test/DeepONet\ PH/v2/pod_train_spatial/checkpoints/LATEST.txt
