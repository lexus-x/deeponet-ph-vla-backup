#!/usr/bin/env bash
# Fit POD trunk basis from LIBERO-Spatial action chunks (CPU), save for training.
set -euo pipefail
CONTRIB="$HOME/Desktop/Ayush PH test/contrib_postjul15"
PY="$HOME/Desktop/Ayush PH test/venv/bin/python"
OUT="$HOME/Desktop/Ayush PH test/DeepONet PH/v2/pod_assets"
mkdir -p "$OUT"
export HF_HOME=/home/user/.cache/huggingface
export HF_TOKEN=$(tr -d '\r\n' < "$HF_HOME/token" || true)
cd "$CONTRIB"
# Fetch chunks if missing (parquet only)
if [[ ! -f data/chunks_spatial.npy ]]; then
  mkdir -p data
  "$PY" fetch_libero_actions.py 2>&1 | tee "$OUT/fetch.log" || true
fi
"$PY" - <<'PY'
import os, sys, torch, numpy as np
sys.path.insert(0, ".")
from pod_trunk import fit_pod, PODHead
out = os.path.expanduser("~/Desktop/Ayush PH test/DeepONet PH/v2/pod_assets")
path = "data/chunks_spatial.npy"
if not os.path.exists(path):
    # synthesize placeholder note
    open(os.path.join(out, "NEED_CHUNKS.txt"),"w").write("run fetch_libero_actions.py first\n")
    print("NO_CHUNKS")
    raise SystemExit(0)
chunks = torch.from_numpy(np.load(path)).float()
print("chunks", chunks.shape)
for p in (16, 32, 64):
    mean, basis, explained, S = fit_pod(chunks, p=p)
    torch.save({"mean": mean, "basis": basis, "explained": explained, "p": p},
               os.path.join(out, f"pod_p{p}.pt"))
    print(f"p={p} var={explained.sum().item():.4f} saved")
print("POD_FIT_DONE", out)
PY
