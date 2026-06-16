#!/usr/bin/env bash
# Multi-seed post-training evaluation of M1/M3/M4 x seeds {0,1,2}:
#   (1) in-distribution closed-loop success on LIBERO-Spatial (20 ep/task)
#   (2) robustness on LIBERO-Plus (7 dims, 15 tasks/dim)
# replan-5 for all. Per-seed invocation; results stream to shared JSONs keyed by
# model name (mN_sS), so everything is resumable.
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate
export TOKENIZERS_PARALLELISM=false

rm -f EVAL_ALL_COMPLETE
for S in 0 1 2; do
  M1=runs/m1_flow_s$S/checkpoints/LATEST
  M3=runs/m3_deeponet_s$S/checkpoints/LATEST
  M4=runs/m4_deeponet_ph_s$S/checkpoints/LATEST

  echo "######## SEED $S : IN-DISTRIBUTION ########"
  MUJOCO_GL=egl python evaluate.py \
    --model m1_s$S=flow=$M1 --model m3_s$S=deeponet=$M3 --model m4_s$S=deeponet=$M4 \
    --suite libero_spatial --only indist --indist_episodes 20 --replan 5 \
    --out runs/eval_indist 2>&1 | tee -a logs/eval_indist.log

  echo "######## SEED $S : LIBERO-Plus ROBUSTNESS ########"
  MUJOCO_GL=egl python evaluate_plus.py \
    --model m1_s$S=flow=$M1 --model m3_s$S=deeponet=$M3 --model m4_s$S=deeponet=$M4 \
    --suite libero_spatial --n_per_cat 15 --replan 5 \
    --out runs/eval_plus 2>&1 | tee -a logs/eval_plus.log
done

python aggregate.py --indist runs/eval_indist/success_rates.json \
  --plus runs/eval_plus/robustness_plus.json --out runs/report 2>&1 | tee logs/aggregate.log

touch EVAL_ALL_COMPLETE
echo "ALL EVAL + AGGREGATION DONE"
