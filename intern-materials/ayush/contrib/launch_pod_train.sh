#!/usr/bin/env bash
# Train POD-trunk DeepONet head on LIBERO-Spatial (head-only stage then full FT).
# Runs after pod_assets/pod_p32.pt exists. Uses blackwell venv + Ayush train.py recipe.
set -euo pipefail
BASE="$HOME/Desktop/Ayush PH test/DeepONet PH/v2"
CONTRIB="$HOME/Desktop/Ayush PH test/contrib_postjul15"
PY="$HOME/Desktop/Ayush PH test/venv/bin/python"
POD="$BASE/pod_assets/pod_p32.pt"
[[ -f "$POD" ]] || { echo "missing $POD — run fit_pod_basis.sh first"; exit 1; }

# Install a tiny adapter that swaps DeepONetHeadV2 for PODHead when DEEPONET_HEAD=pod
cat > "$CONTRIB/patch_pod_into_v2.py" <<'PY'
"""One-shot: patch modeling_smolvla_deeponet_v2 to accept head_type=pod."""
from pathlib import Path
p = Path.home() / "Desktop/Ayush PH test/DeepONet PH/v2/modeling_smolvla_deeponet_v2.py"
text = p.read_text()
if "head_type == \"pod\"" in text:
    print("already patched")
else:
    needle = '        if head_type == "reg":'
    insert = '''        if head_type == "pod":
            import torch, os, sys
            sys.path.insert(0, os.path.expanduser("~/Desktop/Ayush PH test/contrib_postjul15"))
            from pod_trunk import PODHead
            pod = torch.load(os.environ["POD_CKPT"], map_location="cpu", weights_only=False)
            self.deeponet = PODHead(
                context_dim=context_dim, mean=pod["mean"], basis=pod["basis"],
                d_model=d_model, n_queries=n_queries, n_blocks=n_blocks)
        el''' + needle
    if needle not in text:
        raise SystemExit("needle not found")
    p.write_text(text.replace(needle, insert, 1))
    print("patched", p)
PY

"$PY" "$CONTRIB/patch_pod_into_v2.py"

OUT="$BASE/pod_train_spatial"
mkdir -p "$OUT"
export HF_HOME=/home/user/.cache/huggingface
export HF_TOKEN=$(tr -d '\r\n' < "$HF_HOME/token" || true)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export DEEPONET_HEAD=pod POD_CKPT="$POD"
export DEEPONET_P=32 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=0
cd "$BASE"
# Use existing train entrypoint if present
if [[ -f train.py ]]; then
  echo "Launching POD train via train.py (see OUT=$OUT)"
  # Match Ayush Spatial 8.3k-ish budget for a fast first result; extend later
  nohup env PYTHONPATH="$CONTRIB:$BASE" "$PY" train.py \
    --dataset lerobot/libero_spatial_image \
    --out "$OUT" --steps 8300 --batch 48 --seed 0 \
    --head pod \
    >"$OUT/train.log" 2>&1 &
  echo POD_TRAIN_PID=$!
else
  echo "No train.py — write NEED_TRAIN.txt"
  echo "pod assets ready at $POD" > "$OUT/NEED_TRAIN.txt"
fi
