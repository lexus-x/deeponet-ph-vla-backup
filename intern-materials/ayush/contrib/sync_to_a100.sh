#!/usr/bin/env bash
# Sync DeepONet code+ckpts from blackwell -> a100 and run Spatial Plus (M3 vs flow).
set -uo pipefail
SRC_HOST=blackwell
SRC_BASE="/home/user/Desktop/Ayush PH test"
DST="$HOME/deeponet_campaign"
mkdir -p "$DST"

echo "[$(date '+%H:%M:%S')] sync start"
rsync -a --info=progress2 \
  "$SRC_HOST:$SRC_BASE/DeepONet PH/v2/" \
  "$DST/v2/" \
  --exclude 'episode_videos' --exclude 'replan_sweep_results' --exclude 'horizon_sweep_results' \
  --exclude '__pycache__' --exclude '*.mp4' --exclude '*.zip'

# Pull critical checkpoints only
mkdir -p "$DST/ckpts"
rsync -a --info=progress2 \
  "$SRC_HOST:$SRC_BASE/DeepONet PH/v2/deeponet_results/Spatial/runs/m3_s0/checkpoints/30000/" \
  "$DST/ckpts/m3_spatial_30k/"
rsync -a --info=progress2 \
  "$SRC_HOST:$SRC_BASE/DeepONet PH/paper_repro/Spatial/runs/flow_s0/checkpoints/30000/" \
  "$DST/ckpts/flow_spatial_30k/"
rsync -a --info=progress2 \
  "$SRC_HOST:$SRC_BASE/DeepONet PH/v2/deeponet_results/Object/runs/m3_s0/checkpoints/30000/" \
  "$DST/ckpts/m3_object_30k/"
rsync -a --info=progress2 \
  "$SRC_HOST:$SRC_BASE/DeepONet PH/paper_repro/Object/runs/flow_s0/checkpoints/30000/" \
  "$DST/ckpts/flow_object_30k/"
rsync -a --info=progress2 \
  "$SRC_HOST:$SRC_BASE/DeepONet PH/v2/deeponet_results/Long/runs/m3_s0/checkpoints/30000/" \
  "$DST/ckpts/m3_long_30k/"
rsync -a --info=progress2 \
  "$SRC_HOST:$SRC_BASE/DeepONet PH/paper_repro/Long/runs/flow_s0/checkpoints/30000/" \
  "$DST/ckpts/flow_long_30k/"

# contrib modules
rsync -a "$SRC_HOST:$SRC_BASE/contrib_postjul15/" "$DST/contrib/"

echo "[$(date '+%H:%M:%S')] sync done"
ls -la "$DST/ckpts"
