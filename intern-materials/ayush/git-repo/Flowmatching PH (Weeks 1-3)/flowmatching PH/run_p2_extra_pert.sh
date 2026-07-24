#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test"; CODE="$ROOT/flowmatching PH"; HF="$ROOT/hugging face ckp"
cd "$ROOT"; source venv/bin/activate; export MUJOCO_GL=egl
DS=lerobot/libero_object_image; STATS="$HF/ckpt_stats.pt"
for tag in control lambda_0p02 lambda_0p05 lambda_0p1 lambda_0p2 lambda_0p5; do
  md="$HF/ph_object/$tag"
  echo "[p2extra $(date '+%H:%M')] $tag lighting+sensor_noise"
  python "$CODE/evaluate.py" --suite libero_object --dataset $DS --stats_path "$STATS" --skip_baseline \
    --baseline "$md/checkpoints/LATEST" --ph "$md/checkpoints/LATEST" \
    --only liberov --perturbations lighting,sensor_noise --liberov_episodes 6 --out "$md/results" \
    > "logs/p2extra_${tag}.log" 2>&1 && echo "  done $tag" || echo "  FAIL $tag"
done
touch "$ROOT/P2EXTRA_COMPLETE"; echo "[p2extra] ALL DONE"
