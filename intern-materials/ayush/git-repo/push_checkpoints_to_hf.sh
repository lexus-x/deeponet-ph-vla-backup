#!/usr/bin/env bash
# ============================================================================
# Push the 30K paper-repro flow checkpoints (Spatial / Object / Long) to YOUR
# OWN private Hugging Face model repo — the ones NOT covered by
# backup_checkpoints_to_hf.sh. One run does everything.
#
# USAGE:
#   bash push_checkpoints_to_hf.sh                  # prompts for a WRITE token
#   HF_TOKEN=hf_xxx bash push_checkpoints_to_hf.sh  # non-interactive
#
# Get a WRITE token first: https://huggingface.co/settings/tokens (role = Write)
# ============================================================================
set -uo pipefail

ROOT="/home/user/Desktop/Ayush PH test"
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"
command -v hf >/dev/null || { echo "ERROR: 'hf' not found — activate the venv or: pip install -U huggingface_hub"; exit 1; }

# --- 1) token + login ---------------------------------------------------------
TOKEN="${HF_TOKEN:-}"
if [ -z "$TOKEN" ]; then read -rsp "Paste your Hugging Face WRITE token: " TOKEN; echo; fi
[ -n "$TOKEN" ] || { echo "ERROR: no token entered."; exit 1; }
hf auth login --token "$TOKEN" >/dev/null 2>&1 \
  || { echo "ERROR: login failed — token bad/expired or not a WRITE token."; exit 1; }

# --- 2) detect YOUR namespace (prevents the qownscks 403) ---------------------
ME=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null)
[ -n "$ME" ] || { echo "ERROR: could not read your username."; exit 1; }
REPO="$ME/deeponet-ph-vla-checkpoints"
echo ">> logged in as : $ME"
echo ">> target repo  : $REPO  (private)"

# --- 3) create the repo (harmless if it already exists) -----------------------
hf repo create "$REPO" --repo-type model --private 2>/dev/null || true

# --- 4) checkpoints to push:   <local dir> | <path inside the repo> -----------
cd "$ROOT/DeepONet PH"
ITEMS=(
  "paper_repro/Spatial/runs/flow_s0/checkpoints/30000|paper_repro/Spatial_flow_30000"
  "paper_repro/Object/runs/flow_s0/checkpoints/30000|paper_repro/Object_flow_30000"
  "paper_repro/Long/runs/flow_s0/checkpoints/30000|paper_repro/Long_flow_30000"
)

ok=0; bad=0
for it in "${ITEMS[@]}"; do
  IFS='|' read -r src dst <<< "$it"
  if [ ! -f "$src/model.safetensors" ]; then echo "!! SKIP (missing): $src"; bad=$((bad+1)); continue; fi
  echo ">> uploading  $src  ->  $dst"
  if hf upload "$REPO" "$src" "$dst" --repo-type model; then ok=$((ok+1)); else echo "!! FAILED: $src"; bad=$((bad+1)); fi
done

# --- 5) verify ----------------------------------------------------------------
echo; echo ">> paper_repro files now in $REPO:"
python -c "from huggingface_hub import HfApi; print('\n'.join(f for f in HfApi().list_repo_files('$REPO', repo_type='model') if 'paper_repro' in f))"
echo
echo ">> DONE — $ok uploaded, $bad skipped/failed."
echo ">> view: https://huggingface.co/$REPO/tree/main/paper_repro"
