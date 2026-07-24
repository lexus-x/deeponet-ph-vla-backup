#!/usr/bin/env bash
# 2-wide parallel LIBERO-Plus overnight campaign (n_per_cat=8).
# Paths with spaces: never unquoted-expand a combined spec string.
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test"
V2="$ROOT/DeepONet PH/v2"
VENV="$ROOT/venv/bin/python"
OUT="$V2/plus_multisuite_campaign"
mkdir -p "$OUT"
cd "$V2"

export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export DEEPONET_P=256 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=16 DEEPONET_HEAD=deeponet
export PYTHONPATH="$V2:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero_plus}"

N_PER_CAT=8
REPLAN=5
MAX_PARALLEL=2

M3_SP="$V2/deeponet_results/Spatial/runs/m3_s0/checkpoints/30000"
FLOW_SP="$ROOT/DeepONet PH/paper_repro/Spatial/runs/flow_s0/checkpoints/30000"
M3_OBJ="$V2/deeponet_results/Object/runs/m3_s0/checkpoints/30000"
FLOW_OBJ="$ROOT/DeepONet PH/paper_repro/Object/runs/flow_s0/checkpoints/30000"
M3_LONG="$V2/deeponet_results/Long/runs/m3_s0/checkpoints/30000"
FLOW_LONG="$ROOT/DeepONet PH/paper_repro/Long/runs/flow_s0/checkpoints/30000"

LOG="$OUT/parallel_run.log"
echo "PARALLEL_PLUS_START $(date) n_per_cat=$N_PER_CAT" | tee -a "$LOG"

# Remove broken accidental dirs from prior split bug
rm -rf "$OUT/m3_PH" "$OUT/flow_PH" 2>/dev/null || true

launch_one () {
  local tag=$1
  local head=$2
  local ckpt=$3
  local suite=$4
  local odir="$OUT/${tag}_${suite}"
  mkdir -p "$odir"
  if [[ -f "$odir/DONE" ]]; then
    echo "SKIP $tag $suite (DONE)" | tee -a "$LOG"
    return 0
  fi
  if [[ ! -d "$ckpt" ]]; then
    echo "MISSING ckpt: $ckpt — skip $tag $suite" | tee -a "$LOG"
    return 1
  fi
  if [[ -f "$odir/robustness_plus.json" ]]; then
    if TAG="$tag" ODIR="$odir" "$VENV" - <<'PY'
import json, os, sys
d = json.load(open(os.environ["ODIR"] + "/robustness_plus.json"))
m = d.get(os.environ["TAG"], {})
sys.exit(0 if m.get("robustness_average") is not None else 1)
PY
    then
      touch "$odir/DONE"
      echo "SKIP $tag $suite (complete)" | tee -a "$LOG"
      return 0
    fi
  fi
  echo "START $tag $suite ckpt=$ckpt $(date)" | tee -a "$LOG"
  (
    "$VENV" evaluate_plus.py \
      --model "${tag}=${head}=${ckpt}" \
      --suite "$suite" --replan "$REPLAN" --n_per_cat "$N_PER_CAT" \
      --out "$odir" >> "$odir/run.log" 2>&1
    ec=$?
    echo "EXIT $tag $suite code=$ec $(date)" | tee -a "$LOG"
    if [[ $ec -eq 0 ]]; then touch "$odir/DONE"; fi
  ) &
  echo "PID $! -> $tag $suite" | tee -a "$LOG"
}

wait_slot () {
  while true; do
    n=$(jobs -rp | wc -l)
    if [[ "$n" -lt "$MAX_PARALLEL" ]]; then
      break
    fi
    sleep 20
  done
}

run_pair_member () {
  wait_slot
  launch_one "$1" "$2" "$3" "$4"
}

# Object pair
run_pair_member m3 deeponet "$M3_OBJ" libero_object
run_pair_member flow flow "$FLOW_OBJ" libero_object
# Long pair (starts as slots free)
run_pair_member m3 deeponet "$M3_LONG" libero_10
run_pair_member flow flow "$FLOW_LONG" libero_10
# Spatial pair
run_pair_member m3 deeponet "$M3_SP" libero_spatial
run_pair_member flow flow "$FLOW_SP" libero_spatial

echo "ALL_LAUNCHED waiting $(date)" | tee -a "$LOG"
wait
echo "PARALLEL_PLUS_DONE $(date)" | tee -a "$LOG"
