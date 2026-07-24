#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/home/user/.cache/huggingface
PY="/home/user/Desktop/Ayush PH test/venv/bin/python"
mkdir -p "$HF_HOME"
echo "pre-download SmolVLM2..."
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
print("downloading", name, flush=True)
snapshot_download(name)
print("DONE", flush=True)
PY
ls "$HF_HOME/hub" | grep -i SmolVLM || true
echo "probe load..."
cd "/home/user/Desktop/Ayush PH test/DeepONet PH/v2"
export DEEPONET_P=256 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=16 DEEPONET_HEAD=deeponet
"$PY" - <<'PY'
import os, sys, inspect
sys.path.insert(0, ".")
from modeling_smolvla_deeponet_v2 import SmolVLADeepONetPolicy
ckpt = "/home/user/Desktop/Ayush PH test/DeepONet PH/v2/deeponet_results/Long/runs/m3_s0/checkpoints/30000"
p = SmolVLADeepONetPolicy.from_pretrained(ckpt, ph_enabled=False)
print("methods:", [m for m in dir(p) if any(k in m.lower() for k in ("action","predict","select","queue"))])
print(inspect.getsource(p.select_action)[:2000])
PY
