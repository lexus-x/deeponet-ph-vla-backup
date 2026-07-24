#!/usr/bin/env bash
# Internal design ablations of DeepONet-v2 (M3, no PH). Each config removes/limits
# ONE component vs the full v2 (p=256, 3 cross-attn blocks, 8 queries, Fourier-16):
#   p64   : basis p 256 -> 64        (--deeponet_p 64)
#   noF   : Fourier-tau off          (--deeponet_fourier 0)
#   1blk  : 3 -> 1 cross-attn block  (--deeponet_blocks 1)
# 2 seeds each (0,1). Trains, then evals in-dist + LIBERO-Plus with MATCHING arch
# (via DEEPONET_* env vars at load time). Reference = full v2 (already 5 seeds).
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

NAMES=(p64 noF 1blk)
TRAINA=("--deeponet_p 64" "--deeponet_fourier 0" "--deeponet_blocks 1")
ENVV=("DEEPONET_P=64" "DEEPONET_FOURIER=0" "DEEPONET_BLOCKS=1")

COMMON="--dataset lerobot/libero_spatial_image \
  --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 \
  --ckpt_every 10000 --epoch_steps 200 --num_workers 8"

rm -f ABL_DONE
for i in 0 1 2; do
  nm=${NAMES[$i]}; ta=${TRAINA[$i]}; ev=${ENVV[$i]}
  for S in 0 1; do
    echo "###### ablation $nm seed $S (train) ######"
    python train.py --head deeponet --variant baseline --seed $S \
      --out runs/abl_${nm}_s$S $COMMON $ta 2>&1 | tee logs/abl_${nm}_s$S.log
    # prune to final checkpoint
    d=runs/abl_${nm}_s$S; latest=$(cat "$d/checkpoints/LATEST.txt")
    for c in "$d"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done
  done
  echo "###### ablation $nm (eval, env: $ev) ######"
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

python aggregate_ablations.py 2>&1 | tee logs/abl_aggregate.log || echo "abl aggregate failed"
touch ABL_DONE
echo "ABLATIONS DONE"
