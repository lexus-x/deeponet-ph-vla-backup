#!/usr/bin/env bash
# Evaluate official lerobot/smolvla_libero on LIBERO-Object (+ videos + plot).
# All outputs under "hugging face ckp/". Detached; runs alongside the lambda sweep.
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test"; CODE="$ROOT/flowmatching PH"; HF="$ROOT/hugging face ckp"
cd "$ROOT"; source venv/bin/activate; export MUJOCO_GL=egl
mark(){ echo "[hf $(date '+%m-%d %H:%M:%S')] $*"; }
mark "START hf eval (libero_object, official smolvla_libero)"
python "$CODE/eval_hf_ckpt.py" --policy_path "$HF/smolvla_libero_local" \
  --task libero_object --n_episodes 20 --batch_size 1 --render 8 --out "$HF/object" \
  > logs/hf_eval_object.log 2>&1 && mark "done EVAL object" || mark "FAIL EVAL object"
mark "PLOT"
python "$CODE/hf_plot.py" --eval_info "$HF/object/eval_info.json" --out "$HF/object/plots" \
  > logs/hf_plot_object.log 2>&1 && mark "done PLOT" || mark "FAIL PLOT"
mark "ALL HF-EVAL DONE"
touch "$ROOT/HF_EVAL_COMPLETE"
