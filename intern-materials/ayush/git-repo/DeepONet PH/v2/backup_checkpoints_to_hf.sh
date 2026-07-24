#!/usr/bin/env bash
# ============================================================================
# Back up the important checkpoints to a PRIVATE Hugging Face model repo,
# ONE BY ONE, and (optionally) delete each one locally after it uploads — so
# disk frees up as you go.
#
# WHY YOU RUN THIS (not Claude): Claude Code's safety layer blocks the agent
# from bulk-uploading private data to external services. You running it on your
# own machine (already logged in as `qownscks`) is the intended path.
#
# USAGE:
#   cd "/home/user/Desktop/Ayush PH test/DeepONet PH/v2"
#   bash backup_checkpoints_to_hf.sh              # upload only (safe; delete later)
#   DELETE_AFTER=1 bash backup_checkpoints_to_hf.sh   # delete each local copy after a SUCCESSFUL upload
#
# Verify anytime at: https://huggingface.co/qownscks/deeponet-ph-vla-checkpoints/tree/main
# ============================================================================
set -uo pipefail
REPO="qownscks/deeponet-ph-vla-checkpoints"      # <-- change the user/name if you want
DELETE_AFTER="${DELETE_AFTER:-0}"                 # 0 = keep local, 1 = delete after upload OK
ROOT="/home/user/Desktop/Ayush PH test/DeepONet PH"

command -v hf >/dev/null || { echo "ERROR: 'hf' CLI not found. Run: pip install -U huggingface_hub"; exit 1; }
echo ">> using HF account: $(hf auth whoami 2>/dev/null | sed 's/user=//')"
echo ">> creating private repo $REPO (ok if it already exists)"
hf repo create "$REPO" --repo-type model --private 2>/dev/null || true

# Each entry:  <local checkpoint dir> | <path inside the HF repo>
# The whole checkpoint FOLDER is uploaded (model + normalizer/preprocessor + config),
# so each backup is fully loadable later.
MAP=(
  # ---- per-suite FINAL models (15K steps) — the ones behind the per-suite results ----
  "$ROOT/Spatial/runs/flow_s0/checkpoints/15000|Spatial/flow_s0"
  "$ROOT/Spatial/runs/m3_s0/checkpoints/15000|Spatial/m3_s0"
  "$ROOT/Spatial/runs/m4_s0/checkpoints/15000|Spatial/m4_s0"
  "$ROOT/Object/runs/flow_s0/checkpoints/15000|Object/flow_s0"
  "$ROOT/Object/runs/m3_s0/checkpoints/15000|Object/m3_s0"
  "$ROOT/Object/runs/m4_s0/checkpoints/15000|Object/m4_s0"
  "$ROOT/Goal/runs/flow_s0/checkpoints/15000|Goal/flow_s0"
  "$ROOT/Goal/runs/m3_s0/checkpoints/15000|Goal/m3_s0"
  "$ROOT/Goal/runs/m4_s0/checkpoints/15000|Goal/m4_s0"
  # ---- v2 5-seed headline models (8.3K) — the main Spatial comparison ----
  "$ROOT/v2/runs/m3v2_s0/checkpoints/8300|v2/m3v2_s0"
  "$ROOT/v2/runs/m3v2_s1/checkpoints/8300|v2/m3v2_s1"
  "$ROOT/v2/runs/m3v2_s2/checkpoints/8300|v2/m3v2_s2"
  "$ROOT/v2/runs/m3v2_s3/checkpoints/8300|v2/m3v2_s3"
  "$ROOT/v2/runs/m3v2_s4/checkpoints/8300|v2/m3v2_s4"
  "$ROOT/v2/runs/m4v2_s0/checkpoints/8300|v2/m4v2_s0"
  "$ROOT/v2/runs/m4v2_s1/checkpoints/8300|v2/m4v2_s1"
  "$ROOT/v2/runs/m4v2_s2/checkpoints/8300|v2/m4v2_s2"
  "$ROOT/v2/runs/m4v2_s3/checkpoints/8300|v2/m4v2_s3"
  "$ROOT/v2/runs/m4v2_s4/checkpoints/8300|v2/m4v2_s4"
  # ---- (optional) ablation + 5-seed flow baselines — uncomment to also back these up ----
  # "$ROOT/v2/runs/reg_s0/checkpoints/8300|v2/reg_s0"
  # "$ROOT/v2/runs/reg_s1/checkpoints/8300|v2/reg_s1"
  # "$ROOT/v2/runs/abl_noF_s0/checkpoints/8300|v2/abl_noF_s0"
  # "$ROOT/v2/runs/abl_noF_s1/checkpoints/8300|v2/abl_noF_s1"
  # "$ROOT/v2/runs/abl_1blk_s0/checkpoints/8300|v2/abl_1blk_s0"
  # "$ROOT/v2/runs/abl_1blk_s1/checkpoints/8300|v2/abl_1blk_s1"
  # "$ROOT/runs/m1_flow_s0/checkpoints/8300|spatial5seed/m1_flow_s0"
  # "$ROOT/runs/m1_flow_s3/checkpoints/8300|spatial5seed/m1_flow_s3"
  # "$ROOT/runs/m1_flow_s4/checkpoints/8300|spatial5seed/m1_flow_s4"
  # "$ROOT/runs/m3_deeponet_s0/checkpoints/8300|spatial5seed/m3_deeponet_s0"
)

ok=0; fail=0; skip=0
for entry in "${MAP[@]}"; do
  src="${entry%%|*}"; dst="${entry##*|}"
  if [ ! -d "$src" ]; then echo "SKIP (missing): $src"; skip=$((skip+1)); continue; fi
  sz=$(du -sh "$src" 2>/dev/null | cut -f1)
  echo ">> [$((ok+fail+1))] uploading ($sz)  $dst"
  if hf upload "$REPO" "$src" "$dst" --repo-type model; then
    echo "   OK -> $REPO/$dst"
    ok=$((ok+1))
    if [ "$DELETE_AFTER" = "1" ]; then
      echo "   freeing disk: rm -rf $src"
      rm -rf "$src"
    fi
  else
    echo "   FAILED (kept local): $dst"
    fail=$((fail+1))
  fi
done
echo ">> DONE.  uploaded=$ok  failed=$fail  skipped(missing)=$skip"
echo ">> https://huggingface.co/$REPO/tree/main"
[ "$DELETE_AFTER" = "1" ] || echo ">> (local copies kept. Re-run with DELETE_AFTER=1 to free disk after verifying.)"
