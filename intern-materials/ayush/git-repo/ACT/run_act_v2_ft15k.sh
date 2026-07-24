#!/usr/bin/env bash
# ============================================================================
# ACT V2 — RE-FINETUNE at 15K (reuse existing 40-task pretrains; 8K was too short).
#   Pretrains (frozen, reused): $NTFS/runs/pretrain/<variant>/checkpoints/LATEST
#   Outputs (new):              $NTFS/runs_ft15k/  (8K results in runs/ untouched)
#   Per suite x variant: finetune 15K (init_from pretrain) -> in-dist 3-seed + LIBERO-Plus.
#   RUN (detached):
#     setsid nohup bash run_act_v2_ft15k.sh > /media/user/C2FE578FFE577A9D/ACT_v2/run_ft15k.out 2>&1 </dev/null &
# ============================================================================
set -o pipefail
cd "$(dirname "$0")"
ROOT="/home/user/Desktop/Ayush PH test"
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"

NTFS="/media/user/C2FE578FFE577A9D/ACT_v2"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_LEROBOT_HOME="$NTFS/hf_lerobot" HF_HOME="$NTFS/hf_home" HF_HUB_DISABLE_SYMLINKS_WARNING=1

FT_STEPS=${FT_STEPS:-15000}
BATCH=${BATCH:-64}
PRE="$NTFS/runs/pretrain"                 # existing pretrains (reused)
BASE="$NTFS/runs_ft15k"; mkdir -p "$BASE"; PROG="$BASE/PROGRESS.log"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }
guard(){ local kb; kb=$(df --output=avail "$NTFS" | tail -1); [ "$kb" -lt 10485760 ] && { note "ABORT <10GB"; exit 1; } || true; }

NAMES=(Spatial Object Long Goal)
DATA=(lerobot/libero_spatial_image lerobot/libero_object_image lerobot/libero_10_image lerobot/libero_goal_image)
FLAG=(libero_spatial libero_object libero_10 libero_goal)
MAXS=(520 520 520 520)
VARIANTS=(act act_deeponet act_deeponet_ph)
COMMON="--batch $BATCH --ckpt_every 5000 --lr 1e-4 --lr_backbone 1e-5 --ema 0.999 \
  --epoch_steps 200 --num_workers 32 --ph_warmup 5000 --ph_trigger 0.15 --lambda_ph 0.02 --ph_k 8 --seed 0"

python -c "import lerobot, libero, torch; print('env OK cuda=', torch.cuda.is_available())" || { echo "ENV NOT READY"; exit 1; }
for v in "${VARIANTS[@]}"; do [ -f "$PRE/$v/checkpoints/LATEST.txt" ] || { echo "MISSING pretrain $v"; exit 1; }; done

note "==== ACT V2 RE-FINETUNE 15K START (reuse pretrains) ===="
for k in 0 1 2 3; do
  nm=${NAMES[$k]}; ds=${DATA[$k]}; fl=${FLAG[$k]}; ms=${MAXS[$k]}
  dir="$BASE/$nm"; mkdir -p "$dir/runs" "$dir/logs"
  for v in "${VARIANTS[@]}"; do
    guard
    [ -f "$dir/runs/$v/checkpoints/LATEST.txt" ] && { note "FT skip $nm/$v (exists)"; continue; }
    note "FT15K start $nm/$v <- pretrain"
    if python train_act_v2.py --variant "$v" --dataset "$ds" --init_from "$PRE/$v/checkpoints/LATEST" \
         --out "$dir/runs/$v" --steps "$FT_STEPS" $COMMON 2>&1 | tee "$dir/logs/ft_${v}.log"; then
      note "FT15K done  $nm/$v"
    else note "FT15K FAILED $nm/$v"; fi
  done
  guard
  MS=""; for v in "${VARIANTS[@]}"; do [ -f "$dir/runs/$v/checkpoints/LATEST.txt" ] && MS="$MS --model ${v}=$dir/runs/$v/checkpoints/LATEST"; done
  note "EVAL indist start $nm"
  python evaluate_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_indist" \
    --indist_episodes 10 --test_seeds 3 --replan 5 --max_steps "$ms" --only indist \
    2>&1 | tee "$dir/logs/eval_indist.log" && note "EVAL indist done $nm"
  guard
  note "PLUS start $nm"
  python evaluate_plus_lerobot.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_plus" \
    --n_per_cat 12 --replan 5 --max_steps 300 \
    2>&1 | tee "$dir/logs/eval_plus.log" && note "PLUS done $nm"
  note "######## V2-FT15K RESULT READY: $nm ########"
done
note "==== ACT V2 RE-FINETUNE 15K DONE ===="; touch "$BASE/ALL_DONE"
