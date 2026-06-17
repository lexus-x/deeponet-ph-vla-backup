#!/usr/bin/env bash
# =============================================================================
# Live monitor for the BridgeData V2 (bridge_orig RLDS) download.
# Refreshes in place (like `top`) — NOT a one-shot snapshot.
#
#   Usage:  bash watch_bridge.sh           # refresh every 5s
#           bash watch_bridge.sh 3         # refresh every 3s
#   Quit:   Ctrl-C
# =============================================================================
DEST=/media/user/C2FE578FFE577A9D/datasets/bridge_orig_rlds
LOG=/media/user/C2FE578FFE577A9D/datasets/bridge_dl.log
TRAIN_CSV="/home/user/Desktop/Ayush PH test/DeepONet PH/paper_repro/Long/runs/flow_s0/log_step.csv"
TOTAL_MB=135475                 # ~132.3 GB target
INTERVAL=${1:-5}                # refresh seconds (default 5)

prev_mb=0
prev_t=$(date +%s)

trap 'echo; echo "stopped watching (download keeps running in background)."; exit 0' INT

while true; do
  now=$(date +%s)
  pid=$(pgrep -f "hf download shihao1895/bridge-rlds" | head -1)
  mb=$(du -sm "$DEST" 2>/dev/null | cut -f1); mb=${mb:-0}
  shards=$(find "$DEST" -name '*tfrecord*' 2>/dev/null | wc -l)
  pct=$(( mb * 100 / TOTAL_MB )); [ "$pct" -gt 100 ] && pct=100

  dt=$(( now - prev_t )); [ "$dt" -lt 1 ] && dt=1
  rate=$(( (mb - prev_mb) / dt ))           # MB/s since last refresh
  [ "$rate" -lt 0 ] && rate=0
  prev_mb=$mb; prev_t=$now

  clear
  echo "==================== BridgeData V2  (bridge_orig RLDS, 132 GB) ===================="
  echo "  $(date '+%F %T')      refreshing every ${INTERVAL}s      (Ctrl-C to quit)"
  echo "----------------------------------------------------------------------------------"
  if [ -n "$pid" ]; then
    ps -o pid,etime,%cpu,%mem,stat -p "$pid" | sed 's/^/  /'
    echo "  STATUS : DOWNLOADING (pid $pid)"
  else
    if grep -q "download exited" "$LOG" 2>/dev/null; then
      echo "  STATUS : *** FINISHED ***   $(grep 'download exited' "$LOG" | tail -1)"
    else
      echo "  STATUS : process not found (not started, or stopped)"
    fi
  fi
  echo "----------------------------------------------------------------------------------"
  printf "  downloaded : %'d MB / %'d MB   (~%s%%)    shards %s/1024\n" "$mb" "$TOTAL_MB" "$pct" "$shards"
  if [ "$rate" -gt 0 ]; then
    eta=$(( (TOTAL_MB - mb) / rate / 60 ))
    printf "  rate       : ~%s MB/s     ETA ~%s min (~%s h)\n" "$rate" "$eta" "$(awk "BEGIN{printf \"%.1f\", $eta/60}")"
  else
    printf "  rate       : ~0 MB/s     ETA --\n"
  fi
  filled=$(( pct / 2 ))
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
  empty=$(printf '%*s' $(( 50 - filled )) '')
  echo "  [${bar}${empty}] ${pct}%"
  echo "----------------------------------------------------------------------------------"
  root=$(df -h / | tail -1 | awk '{print $4}')
  drive=$(df -h "$DEST" 2>/dev/null | tail -1 | awk '{print $4}')
  step=$(tail -1 "$TRAIN_CSV" 2>/dev/null | cut -d, -f1)
  echo "  root free: ${root}   |   1TB-drive free: ${drive}   |   Long train step: ${step:-?}/30000"
  echo "=================================================================================="

  if [ -z "$pid" ] && grep -q "download exited" "$LOG" 2>/dev/null; then
    echo "  Download complete — Ctrl-C to exit."
  fi
  sleep "$INTERVAL"
done
