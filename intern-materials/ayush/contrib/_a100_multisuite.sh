#!/bin/bash
# Launch Object + Goal in-dist suite comparison on a100 (m3 vs flow) — multi-suite without Plus package.
set -u
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate saptarshi
export MUJOCO_GL=egl
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
CDIR=$HOME/deeponet_campaign
V2=$CDIR/v2
CONTRIB=$CDIR/contrib
OUT=$CDIR/multi_suite_indist
mkdir -p "$OUT"

# Locate ckpts
find_ckpt() {
  local name=$1
  for p in \
    "$CDIR/ckpts/$name/checkpoints/30000" \
    "$CDIR/$name/checkpoints/30000" \
    "$CDIR/$name" \
    "$CDIR/ckpts/$name"
  do
    if [[ -d "$p" ]]; then echo "$p"; return 0; fi
  done
  return 1
}

echo "Looking for ckpts..."
ls -R "$CDIR/ckpts" 2>/dev/null | head -80
ls "$CDIR" | head -40

# Prefer evaluate_exec if present else evaluate.py path used before
EVAL="$CONTRIB/eval_exec_offline.py"
if [[ ! -f "$EVAL" ]]; then EVAL="$V2/evaluate.py"; fi

run_one() {
  local tag=$1 head=$2 ckpt=$3 suite=$4 dataset=$5
  local odir="$OUT/${tag}"
  mkdir -p "$odir"
  if [[ -f "$odir/success_rates.json" ]]; then
    echo "SKIP $tag (exists)"
    return
  fi
  echo "START $tag"
  nohup python "$EVAL" \
    --suite "$suite" --dataset "$dataset" \
    --model "${tag}=${head}=${ckpt}" \
    --exec none --replan 5 --indist_episodes 10 \
    --out "$odir" > "$odir/run.log" 2>&1 &
  echo "pid $! -> $odir"
}

# Discover what we have
M3S=$(find_ckpt m3_spatial_30k || find_ckpt m3_spatial || true)
FLS=$(find_ckpt flow_spatial_30k || find_ckpt flow_spatial || true)
M3O=$(find_ckpt m3_object_30k || find_ckpt m3_object || true)
FLO=$(find_ckpt flow_object_30k || find_ckpt flow_object || true)
M3G=$(find_ckpt m3_goal_30k || find_ckpt m3_goal || true)
FLG=$(find_ckpt flow_goal_30k || find_ckpt flow_goal || true)
M3L=$(find_ckpt m3_long_30k || find_ckpt m3_long || true)
FLL=$(find_ckpt flow_long_30k || find_ckpt flow_long || true)

echo "M3S=$M3S"
echo "FLS=$FLS"
echo "M3O=$M3O FLO=$FLO"
echo "M3G=$M3G FLG=$FLG"
echo "M3L=$M3L FLL=$FLL"

# Spatial already done — skip. Run Object/Goal/Long if ckpts exist.
# Only one job at a time on A100 for EGL stability — chain via wait if needed.
# Launch Object first if available; else Long confirmation at 10 eps.
if [[ -n "${M3O:-}" && -n "${FLO:-}" ]]; then
  run_one m3_object deeponet "$M3O" libero_object lerobot/libero_object_image
  wait
  run_one flow_object flow "$FLO" libero_object lerobot/libero_object_image
  wait
elif [[ -n "${M3L:-}" && -n "${FLL:-}" ]]; then
  # Confirm Long on a100 with EGL (faster than osmesa) — stock baselines only
  run_one m3_long_confirm deeponet "$M3L" libero_10 lerobot/libero_10_image
  wait
  run_one flow_long_confirm flow "$FLL" libero_10 lerobot/libero_10_image
  wait
else
  echo "No Object/Long ckpts found for multi-suite; listing:"
  find "$CDIR" -name 'config.json' 2>/dev/null | head -40
fi

echo "Launched. Watch $OUT"
