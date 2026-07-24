#!/usr/bin/env bash
set -uo pipefail
pgrep -f '_bw_overnight_watcher.sh' >/dev/null && echo watcher_already_up && exit 0
nohup bash /tmp/_bw_overnight_watcher.sh > "/home/user/Desktop/Ayush PH test/DeepONet PH/v2/plus_multisuite_campaign/watcher.log" 2>&1 &
echo WATCHER_PID=$!
sleep 5
pgrep -af '_bw_overnight_watcher' | head -3
head -15 "/home/user/Desktop/Ayush PH test/DeepONet PH/v2/plus_multisuite_campaign/OVERNIGHT_SUMMARY.md"
