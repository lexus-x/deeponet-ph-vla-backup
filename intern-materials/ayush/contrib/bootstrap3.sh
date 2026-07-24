#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/home/user/.cache/huggingface
export HF_TOKEN=$(tr -d '\r\n' < "$HF_HOME/token" || true)
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
PY="/home/user/Desktop/Ayush PH test/venv/bin/python"
echo "resume snapshot_download..."
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
import os
print("token_len", len(os.environ.get("HF_TOKEN", "")), flush=True)
snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
print("DONE", flush=True)
PY
