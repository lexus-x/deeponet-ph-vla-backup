#!/usr/bin/env bash
# ============================================================================
# Goal OBJECT-LAYOUT robustness test — EVAL ONLY on the existing Goal flow ckpt.
# Generalization check: does the 93.5% hold when OBJECT POSITIONS change?
#
# LIBERO has 50 fixed init-state layouts per task. The default eval uses the
# first 20 (init_states[0:20]). We re-run on 3 different layout SLICES:
#   off 0  -> init_states[0:20]   (canonical -> should reproduce ~93.5%, sanity)
#   off 20 -> init_states[20:40]  (different object positions)
#   off 30 -> init_states[30:50]  (different object positions)
# Together these cover all 50 layouts. mean +/- std = robustness to object pos.
#
# Runs CONCURRENTLY with paper_repro_30k (shares GPU). Tiny JSON only, no videos.
# Outputs in a SEPARATE folder (../goal_layouttest).
# ============================================================================
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

BASE=../goal_layouttest
mkdir -p "$BASE/logs"
LOG="$BASE/RUN.log"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
CKPT=../Goal/runs/flow_s0/checkpoints/LATEST

guard(){ local kb; kb=$(df --output=avail / | tail -1)
  if [ "$kb" -lt 10485760 ]; then note "ABORT: <10GB free on /"; exit 1; fi; }

note "==== GOAL OBJECT-LAYOUT TEST START (concurrent, 20 ep/task, replan 5) ===="
note "checkpoint: $CKPT  (= the model that scored 93.5%)"
for off in 0 20 30; do
  guard
  note "LAYOUT off=$off  start  (object layouts init_states[$off:$((off+20))])"
  python evaluate_seedtest.py \
    --model flow=flow=$CKPT \
    --suite libero_goal --dataset lerobot/libero_goal_image \
    --indist_episodes 20 --replan 5 --init_offset $off --base_seed $((1000+off)) \
    --out "$BASE/off_$off" 2>&1 | tee "$BASE/logs/off_$off.log"
  note "LAYOUT off=$off  done"
  python aggregate_goal_seedtest.py 2>&1 | tee -a "$LOG"   # incremental aggregate
done
note "==== ALL 3 LAYOUT SLICES DONE ===="
touch "$BASE/DONE"
