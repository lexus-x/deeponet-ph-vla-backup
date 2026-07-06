#!/usr/bin/env bash
# =============================================================================
# Comp-1 REPLAN=5 re-run: the original comp-1 evals silently used pi0.5's config
# default n_action_steps=50 (nearly open-loop). The SmolVLA study — where DeepONet
# won robustness — used replan=5 (frequent closed-loop re-observation). eval_pi05.py
# is now patched to apply n_action_steps=replan; this re-runs the FULL campaign at
# replan=5 for all three variants, SYMMETRICALLY, so the two studies are comparable.
#
#   flow / deeponet   -> fresh pretrain (seed 0, deterministic) + 4 suites
#   deeponet_ph       -> 4 suites from the PRESERVED pretrain
#
# Results/outputs go to *_replan5 dirs so the original open-loop results are kept
# intact for a side-by-side (replan=50 vs replan=5) comparison.
# Prune-after-eval + GPU guard, exactly like run_c1_reruns.sh.
# Status -> $ROOT/PROGRESS_c1_replan5.log
# =============================================================================
set -uo pipefail
ROOT="/media/user/C2FE578FFE577A9D/sota_campaign"
REPO="/home/user/Desktop/Ayush PH test"
PY="$REPO/venv/bin/python"
PROG="$ROOT/PROGRESS_c1_replan5.log"
export HF_HOME=/media/user/C2FE578FFE577A9D/hf_cache
export HF_LEROBOT_HOME=/media/user/C2FE578FFE577A9D/ACT_v2/hf_lerobot
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TMPDIR="$ROOT/tmp"; mkdir -p "$ROOT/tmp"
ALL4="lerobot/libero_spatial_image,lerobot/libero_object_image,lerobot/libero_10_image,lerobot/libero_goal_image"
declare -A DS=( [Spatial]=lerobot/libero_spatial_image [Object]=lerobot/libero_object_image
                [Long]=lerobot/libero_10_image [Goal]=lerobot/libero_goal_image )
SUITES=(Spatial Object Long Goal)
STEPS=15000; BATCH=16; SEED=0; REPLAN=5
OUTDIR="$ROOT/outputs_replan5"
RESULTS="$ROOT/results_replan5"
mkdir -p "$OUTDIR" "$RESULTS"
PRESERVED_DPH="$ROOT/outputs/_preserved/c1_pi05_deeponet_ph_pretrain_15000"
TRAIN="$REPO/DeepONet PH/pi05/train_pi05.py"
EVAL="$REPO/DeepONet PH/pi05/eval_pi05.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PROG"; }
free_gb(){ df -BG "$ROOT" | awk 'NR==2{gsub("G","",$4);print $4}'; }

finetune_eval(){   # $1 tag  $2 head  $3 variant  $4 suite  $5 pretrain_ckpt
  local tag="$1" head="$2" variant="$3" s="$4" pck="$5" base="$OUTDIR/$1"
  if [ -f "$RESULTS/${tag}__${s}.json" ]; then log "$tag / $s already done -> skip"; return 0; fi
  log "#### $tag : FINETUNE $s  [free $(free_gb)GB] ####"
  "$PY" "$TRAIN" --head "$head" --variant "$variant" --out "$base/$s" \
        --dataset "${DS[$s]}" --init "$pck" \
        --stage1_steps "$STEPS" --stage2_steps 0 --stage1_batch "$BATCH" \
        --ckpt_every "$STEPS" --seed "$SEED" >>"$ROOT/log_r5_${tag}_${s}.out" 2>&1 \
    && log "$tag finetune $s DONE" || { log "$tag finetune $s FAILED"; return 1; }
  log "#### $tag : EVAL $s (replan=$REPLAN, in-dist + LIBERO-Plus) ####"
  "$PY" "$EVAL" --variant "$tag" --head "$head" --ckpt "$base/$s/checkpoints/$STEPS" \
        --suite "$s" --results "$RESULTS" --replan "$REPLAN" --only both \
        >>"$ROOT/log_r5_${tag}_${s}_eval.out" 2>&1 || log "$tag eval $s returned non-zero"
  if [ -f "$RESULTS/${tag}__${s}.json" ]; then
    log "######## $tag / $s RESULT READY (replan=$REPLAN) ########"
    rm -rf "$base/$s/checkpoints"
    log "$tag pruned $s ckpt  [free $(free_gb)GB]"
  else
    log "!!!! $tag / $s produced NO result JSON -> KEEPING ckpt for retry !!!!"
  fi
}

full_variant(){    # $1 tag  $2 head  $3 variant   (fresh pretrain + all 4 suites)
  local tag="$1" head="$2" variant="$3" base="$OUTDIR/$1"
  if [ -d "$base/pretrain/checkpoints/$STEPS" ]; then
    log "$tag pretrain present -> reuse"
  else
    log "#### $tag : PRETRAIN(40 tasks)  [free $(free_gb)GB] ####"
    "$PY" "$TRAIN" --head "$head" --variant "$variant" --out "$base/pretrain" --datasets "$ALL4" \
          --stage1_steps "$STEPS" --stage2_steps 0 --stage1_batch "$BATCH" \
          --ckpt_every "$STEPS" --seed "$SEED" >>"$ROOT/log_r5_${tag}_pretrain.out" 2>&1 \
      && log "$tag pretrain DONE" || { log "$tag pretrain FAILED -> skip variant"; return 1; }
  fi
  local pck="$base/pretrain/checkpoints/$STEPS"
  for s in "${SUITES[@]}"; do finetune_eval "$tag" "$head" "$variant" "$s" "$pck"; done
  rm -rf "$base/pretrain/checkpoints"
  log "==== $tag COMPLETE (replan=$REPLAN)  [free $(free_gb)GB] ===="
}

# ============================ GUARD then RUN =================================
log "==== COMP-1 REPLAN=$REPLAN re-run armed; waiting for any GPU job to release ===="
while pgrep -f "train_pi05.py" >/dev/null || pgrep -f "eval_pi05.py" >/dev/null; do sleep 120; done
log "==== GPU free. Starting replan=$REPLAN full campaign ===="

full_variant   c1_pi05          flow     baseline
full_variant   c1_pi05_deeponet deeponet baseline

# deeponet_ph: all 4 suites from the preserved pretrain
if [ -d "$PRESERVED_DPH" ]; then
  for s in "${SUITES[@]}"; do
    finetune_eval c1_pi05_deeponet_ph deeponet ph "$s" "$PRESERVED_DPH"
  done
  log "==== c1_pi05_deeponet_ph COMPLETE (replan=$REPLAN) ===="
else
  log "!!!! preserved deeponet_ph pretrain missing ($PRESERVED_DPH) -> deeponet_ph SKIPPED !!!!"
fi
log "==== ALL COMP-1 REPLAN=$REPLAN RE-RUNS DONE ===="
