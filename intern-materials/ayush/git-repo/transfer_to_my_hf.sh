#!/usr/bin/env bash
# ============================================================================
# Upload the restored checkpoints to YOUR OWN Hugging Face account.
#
# RUN THIS *AFTER*:
#   (1) the download into ./restored_checkpoints has finished, and
#   (2) you have logged the CLI into YOUR OWN account (NOT the supervisor's):
#         hf auth login        # paste a WRITE token from YOUR account
#
# USAGE:
#   bash transfer_to_my_hf.sh <your-username>/deeponet-ph-vla-checkpoints
# ============================================================================
set -uo pipefail
MYREPO="${1:-}"
[ -n "$MYREPO" ] || { echo "USAGE: bash transfer_to_my_hf.sh <your-username>/deeponet-ph-vla-checkpoints"; exit 1; }
SRC="/home/user/Desktop/Ayush PH test/restored_checkpoints"

who=$(hf auth whoami 2>/dev/null | sed 's/user=//')
echo ">> CLI is logged in as: ${who:-<none>}"
if [ "$who" = "qownscks" ]; then
  echo "STOP: still logged in as the supervisor (qownscks)."
  echo "      Run:  hf auth login    and paste a WRITE token from YOUR OWN account, then re-run."
  exit 1
fi
if [ ! -d "$SRC" ] || [ -z "$(ls -A "$SRC" 2>/dev/null)" ]; then
  echo "STOP: $SRC is missing/empty — wait for the download to finish first."; exit 1
fi

echo ">> creating YOUR private repo: $MYREPO"
hf repo create "$MYREPO" --repo-type model --private 2>/dev/null || true
echo ">> uploading $SRC  ->  $MYREPO"
hf upload "$MYREPO" "$SRC" . --repo-type model
echo ">> DONE. Verify (logged in as you): https://huggingface.co/$MYREPO/tree/main"
