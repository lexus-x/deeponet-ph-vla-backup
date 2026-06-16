#!/usr/bin/env bash
# Option A: lambda sweep for PH on LIBERO-Spatial.
# Reuses the already-trained baseline (lambda=0) and lambda=0.1 PH from the
# strong run. For each new lambda: train PH -> select best ckpt -> eval (PH only).
# Fully detached.
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test"
CODE="$ROOT/flowmatching PH"
OUT="$ROOT/output"
cd "$ROOT"; source venv/bin/activate; export MUJOCO_GL=egl
mkdir -p "$OUT/results_sweep" logs
mark(){ echo "[sweep $(date '+%m-%d %H:%M:%S')] $*"; }
py(){ python "$CODE/$@"; }
DSP=lerobot/libero_spatial_image

mark "START lambda sweep (LIBERO-Spatial)"
for L in 0.02 0.05 0.2 0.5; do
  tag="lambda_${L/./p}"   # e.g. lambda_0p02
  md="$OUT/sweep_spatial_${tag}"
  mark "TRAIN PH spatial lambda=$L"
  py train.py --variant ph --mode full --dataset $DSP --lambda_ph $L \
    --stage1_steps 5000 --stage2_steps 20000 --stage1_batch 16 --stage2_batch 16 \
    --warmup 1500 --num_workers 8 --out "$md" \
    > logs/sweep_train_${tag}.log 2>&1 && mark "done TRAIN $L" || mark "FAIL TRAIN $L"

  mark "SELECT-BEST lambda=$L"
  py select_best.py --model_dir "$md" --suite libero_spatial --dataset $DSP --ph 1 \
    --n_tasks 4 --n_ep 3 --last_k 4 --max_steps 400 \
    > logs/sweep_select_${tag}.log 2>&1 && mark "done SELECT $L" || mark "FAIL SELECT $L"

  mark "EVAL (PH only) lambda=$L"
  py evaluate.py --suite libero_spatial --dataset $DSP --skip_baseline \
    --baseline "$md/checkpoints/BEST" --ph "$md/checkpoints/BEST" \
    --out "$OUT/results_sweep/tmp_${tag}" --libero10_episodes 12 --liberov_episodes 6 \
    > logs/sweep_eval_${tag}.log 2>&1 && mark "done EVAL $L" || mark "FAIL EVAL $L"
  # copy the per-lambda success file to a flat name for the report
  cp "$OUT/results_sweep/tmp_${tag}/success_rates.json" "$OUT/results_sweep/${tag}.json" 2>/dev/null || true
done

mark "REPORT"
py sweep_report.py > logs/sweep_report.log 2>&1 && mark "done REPORT" || mark "FAIL REPORT"
mark "ALL SWEEP DONE"
touch "$ROOT/SWEEP_COMPLETE"
