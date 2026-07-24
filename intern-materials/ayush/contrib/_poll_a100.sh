#!/bin/bash
set -u
echo "=== A100 long ==="
grep 'LIBERO-10' ~/deeponet_campaign/long_confirm_results/run.log | tail -25
ps -ef | grep eval_exec | grep -v grep | head -5
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
echo "=== BW POD ==="
