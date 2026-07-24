#!/usr/bin/env bash
# Method track B: Operator-BID on Long M3 vs stock m3_r5=60% / flow~66%.
# Sharded 2-wide (tasks 0-4 and 5-9) for speed; 10 eps for faster signal.
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test"
V2="$ROOT/DeepONet PH/v2"
CONTRIB="$ROOT/contrib_postjul15"
VENV="$ROOT/venv/bin/python"
CKPT="$V2/deeponet_results/Long/runs/m3_s0/checkpoints/30000"
OUT="$V2/bid_long_campaign"
mkdir -p "$OUT"
export MUJOCO_GL=osmesa TOKENIZERS_PARALLELISM=false
export DEEPONET_HEAD=deeponet DEEPONET_P=256 DEEPONET_FOURIER=16 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8
export PYTHONPATH="$CONTRIB:$V2:${PYTHONPATH:-}"
NORM=policy_preprocessor_step_5_normalizer_processor.safetensors

run_shard () {
  local ids=$1
  local tag=$2
  local odir="$OUT/m3_bid_$tag"
  mkdir -p "$odir"
  echo "START BID shard $tag $(date)" | tee -a "$OUT/run.log"
  nohup env OFFLINE_STATS_SF="$CKPT/$NORM" "$VENV" "$CONTRIB/eval_exec_offline.py" \
    --suite libero_10 --dataset lerobot/libero_10_image \
    --model "m3=deeponet=$CKPT" --exec bid --replan 5 \
    --bid_k 8 --bid_noise 0.02 \
    --indist_episodes 10 --task_ids "$ids" --out "$odir" \
    >> "$odir/run.log" 2>&1 &
  echo "PID $! shard $tag"
}

echo "BID_LONG_START $(date)" | tee "$OUT/run.log"
run_shard "0,1,2,3,4" "t0-4"
run_shard "5,6,7,8,9" "t5-9"
