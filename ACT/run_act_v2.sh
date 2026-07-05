#!/usr/bin/env bash
# ============================================================================
# ACT V2 — LIBERO transfer protocol (40-task pretrain -> per-suite finetune)
#   Stage 1: PRETRAIN each variant on the 40-task union of all 4 suites (15K).
#   Stage 2: FINETUNE on each suite (10 tasks) from the pretrained weights.
#   Eval: in-distribution (3 seeds) + LIBERO-Plus (canonical LeRobot env) per suite.
#   3 variants: act / act_deeponet / act_deeponet_ph.
#   Outputs + HF cache on the NTFS drive (main disk is near-full).
#   RUN (detached):
#     setsid nohup bash run_act_v2.sh > /media/user/C2FE578FFE577A9D/ACT_v2/run.out 2>&1 </dev/null &
# ============================================================================
set -o pipefail
cd "$(dirname "$0")"
ROOT="/home/user/Desktop/Ayush PH test"
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"

NTFS="/media/user/C2FE578FFE577A9D/ACT_v2"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_LEROBOT_HOME="$NTFS/hf_lerobot"          # large LIBERO frame data -> NTFS
export HF_HOME="$NTFS/hf_home"                     # hub metadata -> NTFS
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p "$HF_LEROBOT_HOME" "$HF_HOME"

PRE_STEPS=${PRE_STEPS:-15000}
FT_STEPS=${FT_STEPS:-8000}
BATCH=${BATCH:-64}

BASE="$NTFS/runs"; mkdir -p "$BASE"; PROG="$BASE/PROGRESS.log"
note(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }
guard(){ local kb; kb=$(df --output=avail "$NTFS" | tail -1); [ "$kb" -lt 10485760 ] && { note "ABORT <10GB on NTFS"; exit 1; } || true; }

NAMES=(Spatial Object Long Goal)
DATA=(lerobot/libero_spatial_image lerobot/libero_object_image lerobot/libero_10_image lerobot/libero_goal_image)
FLAG=(libero_spatial libero_object libero_10 libero_goal)
MAXS=(520 520 520 520)
VARIANTS=(act act_deeponet act_deeponet_ph)
UNION="lerobot/libero_spatial_image,lerobot/libero_object_image,lerobot/libero_10_image,lerobot/libero_goal_image"

COMMON="--batch $BATCH --ckpt_every 5000 --lr 1e-4 --lr_backbone 1e-5 --ema 0.999 \
  --epoch_steps 200 --num_workers 32 --ph_warmup 5000 --ph_trigger 0.15 --lambda_ph 0.02 --ph_k 8 --seed 0"

python -c "import lerobot, libero, torch; print('env OK cuda=', torch.cuda.is_available())" \
  || { echo "ENV NOT READY"; exit 1; }

note "==== ACT V2 START (pretrain 40-task ${PRE_STEPS} -> finetune ${FT_STEPS}; 3 variants) ===="

# -------- Stage 1: multi-task pretrain (3 variants) --------
PRE="$BASE/pretrain"; mkdir -p "$PRE/logs"
for v in "${VARIANTS[@]}"; do
  guard
  if [ -f "$PRE/$v/checkpoints/LATEST.txt" ]; then note "PRETRAIN skip $v (exists)"; continue; fi
  note "PRETRAIN start $v on 40-task union (${PRE_STEPS})"
  if python train_act_v2.py --variant "$v" --datasets "$UNION" --out "$PRE/$v" \
       --steps "$PRE_STEPS" $COMMON 2>&1 | tee "$PRE/logs/pretrain_${v}.log"; then
    note "PRETRAIN done  $v"
  else note "PRETRAIN FAILED $v"; fi
done

# -------- Stage 2: per-suite finetune + eval --------
for k in 0 1 2 3; do
  nm=${NAMES[$k]}; ds=${DATA[$k]}; fl=${FLAG[$k]}; ms=${MAXS[$k]}
  dir="$BASE/$nm"; mkdir -p "$dir/runs" "$dir/logs" "$dir/plots"

  for v in "${VARIANTS[@]}"; do
    guard
    init="$PRE/$v/checkpoints/LATEST"
    [ -f "$PRE/$v/checkpoints/LATEST.txt" ] || { note "FT skip $nm/$v (no pretrain)"; continue; }
    if [ -f "$dir/runs/$v/checkpoints/LATEST.txt" ]; then note "FT skip $nm/$v (exists)"; continue; fi
    note "FT start $nm/$v (${FT_STEPS}) <- pretrain"
    if python train_act_v2.py --variant "$v" --dataset "$ds" --init_from "$init" \
         --out "$dir/runs/$v" --steps "$FT_STEPS" $COMMON 2>&1 | tee "$dir/logs/ft_${v}.log"; then
      note "FT done  $nm/$v"
    else note "FT FAILED $nm/$v"; fi
  done

  # in-distribution eval (3 seeds, all variants present)
  guard
  MS=""; for v in "${VARIANTS[@]}"; do [ -f "$dir/runs/$v/checkpoints/LATEST.txt" ] && MS="$MS --model ${v}=$dir/runs/$v/checkpoints/LATEST"; done
  note "EVAL indist start $nm"
  python evaluate_act.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_indist" \
    --indist_episodes 10 --test_seeds 3 --replan 5 --max_steps "$ms" --only indist \
    2>&1 | tee "$dir/logs/eval_indist.log" && note "EVAL indist done $nm"

  # LIBERO-Plus robustness (canonical LeRobot relative-action env)
  guard
  note "PLUS start $nm"
  python evaluate_plus_lerobot.py $MS --suite "$fl" --dataset "$ds" --out "$dir/runs/eval_plus" \
    --n_per_cat 12 --replan 5 --max_steps 300 \
    2>&1 | tee "$dir/logs/eval_plus.log" && note "PLUS done $nm"

  note "######## V2 RESULT READY: $nm ########"
done

note "==== ACT V2 training+eval DONE ===="; touch "$BASE/ALL_DONE"
note "Next: python make_plots_v2.py  (figures from $BASE)"
