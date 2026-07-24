#!/bin/bash
set -u
DST=$HOME/deeponet_campaign
mkdir -p "$DST/long_confirm_results"
nohup bash /tmp/_a100_long_confirm.sh >> "$DST/long_confirm_outer.log" 2>&1 &
echo OUTER_PID=$!
sleep 5
ps -ef | grep -E 'eval_exec|long_confirm' | grep -v grep
tail -40 "$DST/long_confirm_outer.log"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
