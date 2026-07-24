#!/usr/bin/env bash
# Smoke: 1 task x 2 eps for pin mode, then launch full campaign if OK.
set -euo pipefail
BASE="$HOME/Desktop/Ayush PH test/DeepONet PH/v2"
CONTRIB="$HOME/Desktop/Ayush PH test/contrib_postjul15"
PY="$HOME/Desktop/Ayush PH test/venv/bin/python"
cd "$BASE"
export HF_HOME=/home/user/.cache/huggingface
export HF_TOKEN=$(tr -d '\r\n' < /home/user/.cache/huggingface/token 2>/dev/null || true)
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export MUJOCO_GL=osmesa TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export DEEPONET_P=256 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=16 DEEPONET_HEAD=deeponet
export PYTHONPATH="$CONTRIB:$BASE:${PYTHONPATH:-}"

M3="$BASE/deeponet_results/Long/runs/m3_s0/checkpoints/30000"
NORM=policy_preprocessor_step_5_normalizer_processor.safetensors
OUT="$BASE/exec_campaign_results/_smoke_pin"
mkdir -p "$OUT"

echo "[smoke] pin 1-task/2-ep..."
OFFLINE_STATS_SF="$M3/$NORM" "$PY" "$CONTRIB/eval_exec_offline.py" \
  --suite libero_10 --dataset lerobot/libero_10_image \
  --model "m3=deeponet=$M3" --exec pin --replan 5 \
  --indist_episodes 2 --task_ids 0 \
  --out "$OUT" 2>&1 | tee "$OUT/run.log"

"$PY" -c "import json; d=json.load(open('$OUT/success_rates.json')); print('SMOKE_OK', d)"
echo "[smoke] launching full campaign..."
nohup bash "$CONTRIB/run_pinning_campaign.sh" > "$BASE/exec_campaign_results/nohup_campaign.log" 2>&1 &
echo CAMPAIGN_PID=$!
