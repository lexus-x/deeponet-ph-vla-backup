#!/usr/bin/env bash
# PH fine-tune of the official lerobot/smolvla_libero checkpoint on LIBERO-Object,
# across all lambda values, with a flow-only control (without-PH). For each model:
# object accuracy + ONE perturbation (viewpoint) robustness + videos. Then plots
# (accuracy, per-task success, robustness, latency, params) and a summary.
# All outputs under "hugging face ckp/ph_object/". Detached; runs alongside the sweep.
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test"; CODE="$ROOT/flowmatching PH"; HF="$ROOT/hugging face ckp"
cd "$ROOT"; source venv/bin/activate; export MUJOCO_GL=egl
mark(){ echo "[hfph $(date '+%m-%d %H:%M:%S')] $*"; }
py(){ python "$CODE/$@"; }

REMAP="$HF/smolvla_libero_remapped"; STATS="$HF/ckpt_stats.pt"
DS=lerobot/libero_object_image; OUT="$HF/ph_object"; mkdir -p "$OUT"
# gentle continued-finetune LRs to preserve the strong baseline
S2=10000; BB=5e-6; HD=2e-5; WU=500
PERT=viewpoint

# tag  variant  lambda
RUNS=( "control baseline 0.0" "lambda_0p02 ph 0.02" "lambda_0p05 ph 0.05" "lambda_0p1 ph 0.1" "lambda_0p2 ph 0.2" "lambda_0p5 ph 0.5" )

mark "START PH-finetune sweep on official ckpt (object); models: ${#RUNS[@]}"
for r in "${RUNS[@]}"; do
  set -- $r; tag=$1; v=$2; lam=$3; md="$OUT/$tag"
  mark "TRAIN $tag (variant=$v lambda=$lam, continued-finetune $S2 steps bb=$BB hd=$HD)"
  py train.py --variant $v --mode full --base_ckpt "$REMAP" --dataset $DS --stats_path "$STATS" \
    --lambda_ph $lam --stage1_steps 0 --stage2_steps $S2 --stage2_batch 16 \
    --warmup $WU --backbone_lr $BB --head_lr $HD --num_workers 8 --out "$md" \
    > logs/hfph_train_${tag}.log 2>&1 && mark "done TRAIN $tag" || mark "FAIL TRAIN $tag"

  mark "EVAL $tag (object clean + $PERT robustness)"
  py evaluate.py --suite libero_object --dataset $DS --stats_path "$STATS" --skip_baseline \
    --baseline "$md/checkpoints/LATEST" --ph "$md/checkpoints/LATEST" \
    --only both --perturbations $PERT --libero10_episodes 12 --liberov_episodes 6 \
    --out "$md/results" \
    > logs/hfph_eval_${tag}.log 2>&1 && mark "done EVAL $tag" || mark "FAIL EVAL $tag"
done

# ---- videos per lambda (baseline=control vs ph=lambda), object, 2-view ----
CTRL="$OUT/control/checkpoints/LATEST"
for r in "${RUNS[@]}"; do
  set -- $r; tag=$1
  [ "$tag" = "control" ] && continue
  md="$OUT/$tag"
  mark "VIDEOS $tag (control vs $tag, object)"
  py simulate_videos.py --suite libero_object --dataset $DS --stats_path "$STATS" \
    --baseline "$CTRL" --ph "$md/checkpoints/LATEST" --out "$md/videos" --max_steps 300 \
    > logs/hfph_videos_${tag}.log 2>&1 && mark "done VIDEOS $tag" || mark "FAIL VIDEOS $tag"
done

# ---- params + latency (identical across lambdas; compute once) ----
mark "COMPARE (latency + params, once)"
py compare.py --baseline "$OUT/control/checkpoints/LATEST" --ph "$OUT/lambda_0p1/checkpoints/LATEST" \
  --dataset $DS --out "$OUT/_compare" --n 1000 \
  > logs/hfph_compare.log 2>&1 && mark "done COMPARE" || mark "FAIL COMPARE"

# ---- report: per-lambda + cross-lambda plots + summary ----
mark "REPORT"
py hf_ph_report.py --root "$OUT" --compare "$OUT/_compare/compare.json" --pert $PERT \
  > logs/hfph_report.log 2>&1 && mark "done REPORT" || mark "FAIL REPORT"

mark "ALL HF-PH DONE"
touch "$ROOT/HFPH_COMPLETE"
