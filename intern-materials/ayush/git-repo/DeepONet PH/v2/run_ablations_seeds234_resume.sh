#!/usr/bin/env bash
# RESUME of the 3-seed Phase 2 after the prune()-clobbers-$d bug killed it.
# Fix: prune() now uses a LOCAL var, and eval paths use a dedicated loop var (cfg),
# never reused across prune(). p64_s2 is ALREADY trained -> eval only (no retrain).
# Then train+eval noF_s2, 1blk_s2, reg_s2. Results -> ../Ablation_Results/.
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

# NOTE: `local` keeps these from clobbering caller variables (the original bug).
prune(){ local pd=$1; local latest; latest=$(cat "$pd/checkpoints/LATEST.txt")
  for c in "$pd"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done; }

evalcfg(){  # $1=config-dir-prefix (e.g. abl_p64)   $2=env string (e.g. DEEPONET_P=64)
  local cfgd=$1; local ev=$2
  env $ev MUJOCO_GL=egl python evaluate.py \
    --model ${cfgd}_s2=deeponet=runs/${cfgd}_s2/checkpoints/LATEST \
    --suite libero_spatial --only indist --indist_episodes 20 --replan 5 \
    --out runs/eval_abl_indist 2>&1 | tee -a logs/abl_eval_indist.log
  env $ev MUJOCO_GL=egl python evaluate_plus.py \
    --model ${cfgd}_s2=deeponet=runs/${cfgd}_s2/checkpoints/LATEST \
    --suite libero_spatial --n_per_cat 15 --replan 5 \
    --out runs/eval_abl_plus 2>&1 | tee -a logs/abl_eval_plus.log
  rm -rf runs/${cfgd}_s2/checkpoints          # free disk once evaluated
  python aggregate_ablations.py 2>&1 | tail -8 || true
}

rm -f SEEDS234_DONE

# --- p64_s2 already trained: eval only ---
echo "###### eval abl_p64 seed 2 (already trained, salvaged) ######"
evalcfg abl_p64 "DEEPONET_P=64"

# --- remaining configs: train seed 2, then eval ---
DIR=(abl_noF abl_1blk reg)
TA=("--deeponet_fourier 0" "--deeponet_blocks 1" "--deeponet_head reg")
EV=("DEEPONET_FOURIER=0" "DEEPONET_BLOCKS=1" "DEEPONET_HEAD=reg")
for i in 0 1 2; do
  cfg=${DIR[$i]}; ta=${TA[$i]}; ev=${EV[$i]}
  echo "###### $cfg seed 2 ($ta) ######"
  python train.py --head deeponet --variant baseline --seed 2 --out runs/${cfg}_s2 $COMMON $ta \
    2>&1 | tee logs/${cfg}_s2.log
  prune runs/${cfg}_s2
  echo "###### eval $cfg seed 2 (env: $ev) ######"
  evalcfg "$cfg" "$ev"
done

python aggregate_ablations.py 2>&1 | tee logs/abl_aggregate3.log || echo "aggregate failed"
touch SEEDS234_DONE
echo "ALL ABLATIONS 3-SEED DONE"
