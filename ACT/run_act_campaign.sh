#!/usr/bin/env bash
# ============================================================================
# ACT campaign: ACT, ACT+DeepONet, ACT+DeepONet+PH on LIBERO Spatial/Object/Long/Goal.
#   - 30K steps, batch 64, checkpoints every 5K
#   - train 1 seed; TEST 3 seeds (in-distribution) -> mean +/- std
#   - LIBERO-Plus robustness for every model
#   - per-suite plots + comparison + per-episode videos; results after each suite
# Whole model <= 61M params (verified). Outputs in ./act_results/. Progress -> PROGRESS.log
#   RUN (detached):  export MUJOCO_GL=egl
#                    setsid nohup bash run_act_campaign.sh > run.out 2>&1 < /dev/null &
# ============================================================================
set -o pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1

python -c "import lerobot, libero, torch; print('env OK cuda=', torch.cuda.is_available())" \
  || { echo "ENV NOT READY"; exit 1; }

BASE=./act_results; mkdir -p "$BASE"; PROG="$BASE/PROGRESS.log"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }
guard(){ local kb; kb=$(df --output=avail . | tail -1); [ "$kb" -lt 10485760 ] && { note "ABORT <10GB"; exit 1; } || true; }

NAMES=(Spatial Object Long Goal)
DATA=(lerobot/libero_spatial_image lerobot/libero_object_image lerobot/libero_10_image lerobot/libero_goal_image)
FLAG=(libero_spatial libero_object libero_10 libero_goal)
VARIANTS=(act act_deeponet act_deeponet_ph)

COMMON="--steps 30000 --batch 64 --ckpt_every 5000 --lr 1e-4 --lr_backbone 1e-5 \
  --ema 0.999 --epoch_steps 200 --num_workers 32 --ph_warmup 5000 --ph_trigger 0.15 --lambda_ph 0.02 --ph_k 8 --seed 0"
INDIST="--indist_episodes 10 --test_seeds 3 --replan 5 --max_steps 520 --only indist"
PLUS="--n_per_cat 12 --replan 5 --max_steps 300"

note "==== ACT campaign START (30K, batch64; Spatial/Object/Long/Goal; 3 variants) ===="
for k in 0 1 2 3; do
  nm=${NAMES[$k]}; ds=${DATA[$k]}; fl=${FLAG[$k]}
  dir="$BASE/$nm"; mkdir -p "$dir/runs" "$dir/logs" "$dir/plots"

  # ---- train the 3 variants ----
  for v in "${VARIANTS[@]}"; do
    guard
    note "TRAIN start  $nm/$v 30K"
    if python train_act.py --variant "$v" --dataset "$ds" --out "$dir/runs/${v}" $COMMON \
         2>&1 | tee "$dir/logs/train_${v}.log"; then note "TRAIN done   $nm/$v"
    else note "TRAIN FAILED $nm/$v"; fi
  done

  # ---- in-distribution eval (all 3, 3-seed) ----
  guard
  MS=""; for v in "${VARIANTS[@]}"; do [ -f "$dir/runs/${v}/checkpoints/LATEST.txt" ] && MS="$MS --model ${v}=$dir/runs/${v}/checkpoints/LATEST"; done
  note "EVAL  start  $nm in-dist (3-seed)"
  python evaluate_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_indist" $INDIST \
    2>&1 | tee "$dir/logs/eval_indist.log" && note "EVAL  done   $nm in-dist"

  # ---- LIBERO-Plus robustness (all 3) ----
  guard
  note "PLUS  start  $nm LIBERO-Plus"
  python evaluate_plus_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_plus" $PLUS \
    2>&1 | tee "$dir/logs/eval_plus.log" && note "PLUS  done   $nm"

  # ---- comparison plot + table ----
  python compare_act.py --suite "$nm" \
    --indist "$dir/runs/eval_indist/success_rates.json" \
    --plus "$dir/runs/eval_plus/success_rates.json" \
    --out_dir "$dir/plots" 2>&1 | tee "$dir/logs/compare_${nm}.log" || note "compare failed $nm"
  note "######## RESULT READY: $nm -> $dir (in-dist 3-seed + LIBERO-Plus + plots) ########"
done
note "==== ACT campaign DONE ===="; touch "$BASE/ALL_DONE"
