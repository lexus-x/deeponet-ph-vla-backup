#!/usr/bin/env bash
# Regression-head baseline (the decisive "operator vs any single-pass head" test).
# Same cross-attention context as DeepONet-v2, but a plain MLP -> action chunk
# (no branch/trunk operator), matched ~10.4M params. 2 seeds. WAITS for the
# current internal ablations (ABL_DONE) to avoid concurrent-training OOM.
# Results go to the SEPARATE ../Ablation_Results/ folder.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

until [ -f ABL_DONE ]; do sleep 300; done   # don't OOM against the internal ablations

COMMON="--dataset lerobot/libero_spatial_image \
  --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 \
  --ckpt_every 10000 --epoch_steps 200 --num_workers 8"

rm -f REG_DONE
for S in 0 1; do
  echo "###### regression-head baseline seed $S (train) ######"
  python train.py --head deeponet --variant baseline --deeponet_head reg --seed $S \
    --out runs/reg_s$S $COMMON 2>&1 | tee logs/reg_s$S.log
  d=runs/reg_s$S; latest=$(cat "$d/checkpoints/LATEST.txt")
  for c in "$d"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done
done

echo "###### regression-head baseline (eval, DEEPONET_HEAD=reg) ######"
env DEEPONET_HEAD=reg MUJOCO_GL=egl python evaluate.py \
  --model reg_s0=deeponet=runs/reg_s0/checkpoints/LATEST \
  --model reg_s1=deeponet=runs/reg_s1/checkpoints/LATEST \
  --suite libero_spatial --only indist --indist_episodes 20 --replan 5 \
  --out runs/eval_abl_indist 2>&1 | tee -a logs/abl_eval_indist.log
env DEEPONET_HEAD=reg MUJOCO_GL=egl python evaluate_plus.py \
  --model reg_s0=deeponet=runs/reg_s0/checkpoints/LATEST \
  --model reg_s1=deeponet=runs/reg_s1/checkpoints/LATEST \
  --suite libero_spatial --n_per_cat 15 --replan 5 \
  --out runs/eval_abl_plus 2>&1 | tee -a logs/abl_eval_plus.log

# refresh the ablation table (now includes the regression-head row)
python aggregate_ablations.py 2>&1 | tee logs/abl_aggregate.log || echo "aggregate failed"
touch REG_DONE
echo "REGRESSION-HEAD BASELINE DONE"
