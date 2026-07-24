#!/usr/bin/env bash
# Full post-training pipeline: waits for training, then compare -> videos -> eval -> plots.
# Fully detached; survives terminal/IDE/folder close.
set -uo pipefail
cd "/home/user/Desktop/Ayush PH test"
source venv/bin/activate
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=0
mark(){ echo "[pipeline $(date '+%H:%M:%S')] $*"; }

BASE=outputs/baseline_full/checkpoints/LATEST
PH=outputs/ph_full/checkpoints/LATEST

mark "waiting for training to finish ..."
while ! grep -q "ALL TRAINING DONE" logs/training_full.log 2>/dev/null; do sleep 60; done
mark "training done. starting post-training pipeline."

mark "STEP 1/4: compare.py (params + latency, 1000 passes)"
python compare.py --baseline "$BASE" --ph "$PH" --out results --n 1000 \
    > logs/compare.log 2>&1 && mark "compare done" || mark "compare FAILED (see logs/compare.log)"

mark "STEP 2/4: simulate_videos.py (~51 MP4s)"
python simulate_videos.py --baseline "$BASE" --ph "$PH" --out videos \
    > logs/videos.log 2>&1 && mark "videos done" || mark "videos FAILED (see logs/videos.log)"

mark "STEP 3/4: evaluate.py (LIBERO-10 12ep + LIBERO-V 8ep; the long step)"
python evaluate.py --baseline "$BASE" --ph "$PH" --out results \
    --libero10_episodes 12 --liberov_episodes 8 \
    > logs/eval.log 2>&1 && mark "eval done" || mark "eval FAILED (see logs/eval.log)"

mark "STEP 4/4: plots.py (6 figures PNG+PDF + summary.pdf)"
python plots.py > logs/plots.log 2>&1 && mark "plots done" || mark "plots FAILED (see logs/plots.log)"

mark "ALL PIPELINE DONE"
touch PIPELINE_COMPLETE
