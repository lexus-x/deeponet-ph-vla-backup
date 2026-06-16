#!/usr/bin/env bash
# Full 5-seed evaluation of the main comparison:
#   v2 models (M3-v2, M4-v2, seeds 0-4) -> runs/eval_v2_{indist,plus}
#   M1 flow seeds 3-4 -> parent runs/eval_{indist,plus} (joins existing s0-2)
# Resumable throughout.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export TOKENIZERS_PARALLELISM=false

rm -f V2_EVAL_DONE
# --- v2 in-distribution (5 seeds) ---
for S in 0 1 2 3 4; do
  MUJOCO_GL=egl python evaluate.py \
    --model m3v2_s$S=deeponet=runs/m3v2_s$S/checkpoints/LATEST \
    --model m4v2_s$S=deeponet=runs/m4v2_s$S/checkpoints/LATEST \
    --suite libero_spatial --only indist --indist_episodes 20 --replan 5 \
    --out runs/eval_v2_indist 2>&1 | tee -a logs/eval_v2_indist.log
done
# --- M1 flow seeds 3-4 in-dist (into parent JSON) ---
for S in 3 4; do
  MUJOCO_GL=egl python evaluate.py \
    --model m1_s$S=flow=../runs/m1_flow_s$S/checkpoints/LATEST \
    --suite libero_spatial --only indist --indist_episodes 20 --replan 5 \
    --out ../runs/eval_indist 2>&1 | tee -a logs/eval_m1_s${S}_indist.log
done
# --- v2 robustness (5 seeds) ---
for S in 0 1 2 3 4; do
  MUJOCO_GL=egl python evaluate_plus.py \
    --model m3v2_s$S=deeponet=runs/m3v2_s$S/checkpoints/LATEST \
    --model m4v2_s$S=deeponet=runs/m4v2_s$S/checkpoints/LATEST \
    --suite libero_spatial --n_per_cat 15 --replan 5 \
    --out runs/eval_v2_plus 2>&1 | tee -a logs/eval_v2_plus.log
done
# --- M1 flow seeds 3-4 robustness (into parent JSON) ---
for S in 3 4; do
  MUJOCO_GL=egl python evaluate_plus.py \
    --model m1_s$S=flow=../runs/m1_flow_s$S/checkpoints/LATEST \
    --suite libero_spatial --n_per_cat 15 --replan 5 \
    --out ../runs/eval_plus 2>&1 | tee -a logs/eval_m1_s${S}_plus.log
done
touch V2_EVAL_DONE
echo "V2 5-SEED EVAL DONE"
