#!/bin/bash
nohup bash -c 'while pgrep -f "pod_train_spatial" >/dev/null; do sleep 60; done; echo TRAIN_DONE $(date); bash /tmp/_bw_pod_eval.sh' > /tmp/pod_eval_watchdog.log 2>&1 &
echo WATCHDOG_PID=$!
pgrep -af pod_train_spatial | head -3
tail -3 "/home/user/Desktop/Ayush PH test/DeepONet PH/v2/pod_train_spatial/train.log"
