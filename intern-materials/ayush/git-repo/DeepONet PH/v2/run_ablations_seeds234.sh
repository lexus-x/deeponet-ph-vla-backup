#!/usr/bin/env bash
# Add seed 2 to ALL ablation configs (p64, noF, 1blk, reg) -> 3 seeds total (0,1,2).
# (User reduced 5 -> 3 seeds.) WAITS for the 2-seed run (REG_DONE) so it doesn't OOM.
# Per-config: train seed 2, eval it (matching arch via env), delete that checkpoint
# to save disk, then re-aggregate. Results -> ../Ablation_Results/.
set -e
set -o pipefail
cd "$(dirname "$0")"
source ../../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

until [ -f REG_DONE ]; do sleep 300; done   # let the 2-seed run finish first

COMMON="--dataset lerobot/libero_spatial_image \
  --stage1_steps 1650 --stage2_steps 6650 --stage1_batch 48 --stage2_batch 48 \
  --warmup 500 --head_lr 1e-4 --backbone_lr 1e-5 --ema 0.999 \
  --ckpt_every 10000 --epoch_steps 200 --num_workers 8"

DIR=(abl_p64 abl_noF abl_1blk reg)
TA=("--deeponet_p 64" "--deeponet_fourier 0" "--deeponet_blocks 1" "--deeponet_head reg")
EV=("DEEPONET_P=64" "DEEPONET_FOURIER=0" "DEEPONET_BLOCKS=1" "DEEPONET_HEAD=reg")

prune(){ d=$1; latest=$(cat "$d/checkpoints/LATEST.txt"); for c in "$d"/checkpoints/*/; do [ "$(basename "$c")" != "$latest" ] && rm -rf "$c" || true; done; }

rm -f SEEDS234_DONE
for i in 0 1 2 3; do
  d=${DIR[$i]}; ta=${TA[$i]}; ev=${EV[$i]}
  S=2
  echo "###### $d seed $S ($ta) ######"
  python train.py --head deeponet --variant baseline --seed $S --out runs/${d}_s$S $COMMON $ta \
    2>&1 | tee logs/${d}_s$S.log
  prune runs/${d}_s$S
  echo "###### eval $d seed 2 (env: $ev) ######"
  env $ev MUJOCO_GL=egl python evaluate.py \
    --model ${d}_s2=deeponet=runs/${d}_s2/checkpoints/LATEST \
    --suite libero_spatial --only indist --indist_episodes 20 --replan 5 \
    --out runs/eval_abl_indist 2>&1 | tee -a logs/abl_eval_indist.log
  env $ev MUJOCO_GL=egl python evaluate_plus.py \
    --model ${d}_s2=deeponet=runs/${d}_s2/checkpoints/LATEST \
    --suite libero_spatial --n_per_cat 15 --replan 5 \
    --out runs/eval_abl_plus 2>&1 | tee -a logs/abl_eval_plus.log
  # free disk: checkpoint no longer needed once evaluated
  rm -rf runs/${d}_s2/checkpoints
  python aggregate_ablations.py 2>&1 | tail -8 || true   # refresh after each config
done

python aggregate_ablations.py 2>&1 | tee logs/abl_aggregate3.log || echo "aggregate failed"
touch SEEDS234_DONE
echo "ALL ABLATIONS 3-SEED DONE"
