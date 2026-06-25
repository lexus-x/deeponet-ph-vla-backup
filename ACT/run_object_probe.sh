#!/usr/bin/env bash
# OBJECT PROBE — test whether MILD, colour-only augmentation (no geometric crop)
# lifts Object-Plus off the floor WITHOUT the collateral damage the strong aug caused.
# Trains all 3 Object variants (reusable in a full campaign if the probe wins), runs a
# quick in-dist sanity gate + the FULL Plus protocol (identical to the original eval so
# the comparison is fair). Writes to a SEPARATE folder; originals untouched.
# Augmentation is applied identically to ACT and both DeepONet variants (training only).
set -u
cd "/home/user/Desktop/Ayush PH test/ACT"
source ../venv/bin/activate
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
PROBE=./act_results_aug_probe
dir="$PROBE/Object"
PROG="$PROBE/PROBE_PROGRESS.log"
mkdir -p "$dir/runs" "$dir/logs"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }
guard(){ local kb; kb=$(df --output=avail . | tail -1); [ "$kb" -lt 10485760 ] && { note "ABORT <10GB free"; exit 1; } || true; }

ds=lerobot/libero_object_image
fl=libero_object
VARIANTS=(act act_deeponet act_deeponet_ph)
COMMON="--steps 30000 --batch 64 --ckpt_every 30000 --lr 1e-4 --lr_backbone 1e-5 \
  --ema 0.999 --epoch_steps 200 --num_workers 32 --ph_warmup 5000 --ph_trigger 0.15 \
  --lambda_ph 0.02 --ph_k 8 --seed 0 --augment --aug_mode mild"

note "==== OBJECT PROBE START (mild colour-only aug, no crop) ===="
for v in "${VARIANTS[@]}"; do
  guard; note "TRAIN+mildaug start Object/$v"
  if python train_act.py --variant "$v" --dataset "$ds" --out "$dir/runs/${v}" $COMMON \
       2>&1 | tee "$dir/logs/train_${v}.log"; then note "TRAIN+mildaug done Object/$v"
  else note "TRAIN+mildaug FAIL Object/$v"; fi
done

MS=""; for v in "${VARIANTS[@]}"; do MS="$MS --model ${v}=$dir/runs/${v}/checkpoints/LATEST"; done

# quick in-dist sanity gate (did mild aug tank in-dist? 5 ep / 1 seed is enough to tell)
guard; note "INDIST(quick) start Object"
python evaluate_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_indist_quick" \
  --indist_episodes 5 --test_seeds 1 --replan 5 --only indist --max_steps 520 \
  2>&1 | tee "$dir/logs/eval_indist_quick.log" && note "INDIST(quick) done Object"

# full Plus — IDENTICAL protocol to the original eval (n_per_cat 12, replan 5, max_steps 300)
guard; note "PLUS start Object"
python evaluate_plus_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_plus" \
  --n_per_cat 12 --replan 5 --max_steps 300 \
  2>&1 | tee "$dir/logs/eval_plus.log" && note "PLUS done Object"

note "######## OBJECT PROBE RESULT READY -> $dir ########"; touch "$PROBE/PROBE_DONE"
