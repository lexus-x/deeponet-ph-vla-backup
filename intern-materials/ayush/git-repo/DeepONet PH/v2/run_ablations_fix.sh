#!/usr/bin/env bash
# FIXED ablation run. train.py default deeponet_p is now 256, so noF/1blk train at
# the correct p=256 (only their own component ablated). p64 result is already
# valid in the eval JSONs (kept). Then the regression-head baseline. All -> the
# separate ../Ablation_Results/ folder.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

COMMON="--dataset lerobot/libero_spatial_image \
  --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 \
  --ckpt_every 10000 --epoch_steps 200 --num_workers 8"

prune(){ d=$1; latest=$(cat "$d/checkpoints/LATEST.txt"); for c in "$d"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done; }

rm -f ABL_DONE REG_DONE
NAMES=(noF 1blk); TRAINA=("--deeponet_fourier 0" "--deeponet_blocks 1"); ENVV=("DEEPONET_FOURIER=0" "DEEPONET_BLOCKS=1")
for i in 0 1; do
  nm=${NAMES[$i]}; ta=${TRAINA[$i]}; ev=${ENVV[$i]}
  for S in 0 1; do
    echo "###### ablation $nm seed $S (p=256, $ta) ######"
    python train.py --head deeponet --variant baseline --seed $S --out runs/abl_${nm}_s$S $COMMON $ta \
      2>&1 | tee logs/abl_${nm}_s$S.log
    prune runs/abl_${nm}_s$S
  done
  echo "###### eval $nm (env: $ev) ######"
  env $ev MUJOCO_GL=egl python evaluate.py \
    --model abl_${nm}_s0=deeponet=runs/abl_${nm}_s0/checkpoints/LATEST \
    --model abl_${nm}_s1=deeponet=runs/abl_${nm}_s1/checkpoints/LATEST \
    --suite libero_spatial --only indist --indist_episodes 20 --replan 5 \
    --out runs/eval_abl_indist 2>&1 | tee -a logs/abl_eval_indist.log
  env $ev MUJOCO_GL=egl python evaluate_plus.py \
    --model abl_${nm}_s0=deeponet=runs/abl_${nm}_s0/checkpoints/LATEST \
    --model abl_${nm}_s1=deeponet=runs/abl_${nm}_s1/checkpoints/LATEST \
    --suite libero_spatial --n_per_cat 15 --replan 5 \
    --out runs/eval_abl_plus 2>&1 | tee -a logs/abl_eval_plus.log
done
touch ABL_DONE

echo "###### regression-head baseline (matched ~10.4M, no operator) ######"
for S in 0 1; do
  python train.py --head deeponet --variant baseline --deeponet_head reg --seed $S \
    --out runs/reg_s$S $COMMON 2>&1 | tee logs/reg_s$S.log
  prune runs/reg_s$S
done
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

python aggregate_ablations.py 2>&1 | tee logs/abl_aggregate.log || echo "aggregate failed"
touch REG_DONE
echo "ALL ABLATIONS + REG BASELINE DONE"
