#!/usr/bin/env bash
# Re-run ONLY the LIBERO-Plus robustness eval (in-dist already complete).
# pipefail so a python crash propagates through `tee` and aborts before the marker.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../venv/bin/activate
export TOKENIZERS_PARALLELISM=false

rm -f runs/eval_plus/robustness_plus.json EVAL_ALL_COMPLETE
for S in 0 1 2; do
  M1=runs/m1_flow_s$S/checkpoints/LATEST
  M3=runs/m3_deeponet_s$S/checkpoints/LATEST
  M4=runs/m4_deeponet_ph_s$S/checkpoints/LATEST
  echo "######## SEED $S : LIBERO-Plus ########"
  MUJOCO_GL=egl python evaluate_plus.py \
    --model m1_s$S=flow=$M1 --model m3_s$S=deeponet=$M3 --model m4_s$S=deeponet=$M4 \
    --suite libero_spatial --n_per_cat 15 --replan 5 \
    --out runs/eval_plus 2>&1 | tee -a logs/eval_plus2.log
done

python aggregate.py --indist runs/eval_indist/success_rates.json \
  --plus runs/eval_plus/robustness_plus.json --out runs/report 2>&1 | tee logs/aggregate.log

touch EVAL_ALL_COMPLETE
echo "PLUS EVAL + AGGREGATION DONE"
