#!/usr/bin/env bash
# STRONG run (max-accuracy): 2 suites (LIBERO-10 + LIBERO-Spatial) x (baseline + PH).
# batch 16, ~6-epoch Stage 2, then BEST-checkpoint selection, full eval, 2-view
# videos, plots. Code lives in "flowmatching PH/"; ALL outputs go to "output/".
# Fully detached; survives terminal/IDE/folder close.
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test"
CODE="$ROOT/flowmatching PH"
OUT="$ROOT/output"
cd "$ROOT"
source venv/bin/activate
export MUJOCO_GL=egl
mkdir -p "$OUT" logs
mark(){ echo "[strong $(date '+%m-%d %H:%M:%S')] $*"; }

D10=lerobot/libero_10_image
DSP=lerobot/libero_spatial_image
py(){ python "$CODE/$@"; }

mark "START max-accuracy strong run"

# ---------- TRAIN 4 models (batch 16, ~6 epochs stage2) ----------
# tag  dataset  stage2_steps
for spec in "libero10 $D10 38000" "spatial $DSP 20000"; do
  set -- $spec; tag=$1; ds=$2; s2=$3
  for v in baseline ph; do
    mark "TRAIN $tag $v (stage1=5000 stage2=$s2 bs16 ~6ep)"
    py train.py --variant $v --mode full --dataset $ds \
      --stage1_steps 5000 --stage2_steps $s2 --stage1_batch 16 --stage2_batch 16 \
      --warmup 1500 --num_workers 8 --out "$OUT/strong_${tag}_${v}" \
      > logs/strong_train_${tag}_${v}.log 2>&1 && mark "done TRAIN $tag $v" || mark "FAIL TRAIN $tag $v"
  done
done

# ---------- BEST-checkpoint selection (maximize accuracy) ----------
sel(){ # tag dataset suite variant ph
  local tag=$1 ds=$2 suite=$3 v=$4 ph=$5
  mark "SELECT-BEST $tag $v"
  py select_best.py --model_dir "$OUT/strong_${tag}_${v}" --suite $suite --dataset $ds \
    --ph $ph --n_tasks 4 --n_ep 3 --last_k 4 --max_steps 400 \
    > logs/strong_select_${tag}_${v}.log 2>&1 && mark "done SELECT $tag $v" || mark "FAIL SELECT $tag $v"
}
sel libero10 $D10 libero_10      baseline 0
sel libero10 $D10 libero_10      ph       1
sel spatial  $DSP libero_spatial baseline 0
sel spatial  $DSP libero_spatial ph       1

# ---------- COMPARE (params + latency) ----------
mark "COMPARE"
py compare.py \
  --baseline "$OUT/strong_libero10_baseline/checkpoints/BEST" \
  --ph       "$OUT/strong_libero10_ph/checkpoints/BEST" \
  --out "$OUT/results_strong" --n 1000 \
  > logs/strong_compare.log 2>&1 && mark "done COMPARE" || mark "FAIL COMPARE"

# ---------- EVAL both suites on BEST (12 in-dist, 6 perturbed) ----------
mark "EVAL libero_10"
py evaluate.py --suite libero_10 --dataset $D10 \
  --baseline "$OUT/strong_libero10_baseline/checkpoints/BEST" \
  --ph       "$OUT/strong_libero10_ph/checkpoints/BEST" \
  --out "$OUT/results_strong" --libero10_episodes 12 --liberov_episodes 6 \
  > logs/strong_eval_libero10.log 2>&1 && mark "done EVAL libero_10" || mark "FAIL EVAL libero_10"

mark "EVAL libero_spatial"
py evaluate.py --suite libero_spatial --dataset $DSP \
  --baseline "$OUT/strong_spatial_baseline/checkpoints/BEST" \
  --ph       "$OUT/strong_spatial_ph/checkpoints/BEST" \
  --out "$OUT/results_strong" --libero10_episodes 12 --liberov_episodes 6 \
  > logs/strong_eval_spatial.log 2>&1 && mark "done EVAL libero_spatial" || mark "FAIL EVAL libero_spatial"

# ---------- VIDEOS both suites on BEST (2-view: agentview|wrist) ----------
mark "VIDEOS libero_10"
py simulate_videos.py --suite libero_10 --dataset $D10 \
  --baseline "$OUT/strong_libero10_baseline/checkpoints/BEST" \
  --ph       "$OUT/strong_libero10_ph/checkpoints/BEST" \
  --out "$OUT/videos_strong/libero10" \
  > logs/strong_videos_libero10.log 2>&1 && mark "done VIDEOS libero_10" || mark "FAIL VIDEOS libero_10"

mark "VIDEOS libero_spatial"
py simulate_videos.py --suite libero_spatial --dataset $DSP \
  --baseline "$OUT/strong_spatial_baseline/checkpoints/BEST" \
  --ph       "$OUT/strong_spatial_ph/checkpoints/BEST" \
  --out "$OUT/videos_strong/spatial" \
  > logs/strong_videos_spatial.log 2>&1 && mark "done VIDEOS libero_spatial" || mark "FAIL VIDEOS libero_spatial"

# ---------- PLOTS + summary ----------
mark "PLOTS"
py plots.py --results "$OUT/results_strong/success_rates.json" --compare "$OUT/results_strong/compare.json" \
  --out "$OUT/plots_strong" --summary "$OUT/summary_strong.pdf" \
  --runs "LIBERO-10:baseline=$OUT/strong_libero10_baseline,LIBERO-10:ph=$OUT/strong_libero10_ph,LIBERO_SPATIAL:baseline=$OUT/strong_spatial_baseline,LIBERO_SPATIAL:ph=$OUT/strong_spatial_ph" \
  > logs/strong_plots.log 2>&1 && mark "done PLOTS" || mark "FAIL PLOTS"

mark "ALL STRONG DONE"
touch "$ROOT/STRONG_COMPLETE"
