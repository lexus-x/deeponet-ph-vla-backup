#!/usr/bin/env bash
set -uo pipefail
cd "/home/user/Desktop/Ayush PH test"
source venv/bin/activate
export MUJOCO_GL=egl
echo "===== BASELINE full training start $(date) ====="
python train.py --variant baseline --mode full --num_workers 8 --out outputs/baseline_full
echo "===== PH full training start $(date) ====="
python train.py --variant ph --mode full --num_workers 8 --out outputs/ph_full
echo "===== ALL TRAINING DONE $(date) ====="
