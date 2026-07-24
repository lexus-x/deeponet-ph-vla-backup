#!/usr/bin/env bash
# UNATTENDED FINALIZER. Waits for the 5-seed eval to finish, then builds the full
# deliverable with NO further human/agent input: final 5-seed report + plots,
# in-distribution videos (all 10 tasks, flow|v1|v2), perturbed videos, and a
# zipped DeepONet_Results/ folder for the presentation.
#
# Deliberately NOT `set -e`: each step is best-effort so one failure can't wipe
# out the rest of the deliverable.
cd "$(dirname "$0")"                       # .../DeepONet PH
source ../venv/bin/activate
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
LOG=logs/finalize.log
mkdir -p logs runs
echo "[finalize] waiting for v2/V2_EVAL_DONE ..." | tee -a "$LOG"
until [ -f v2/V2_EVAL_DONE ]; do sleep 300; done
echo "[finalize] eval done -> building deliverable $(date)" | tee -a "$LOG"

# 1) final 5-seed report + plots
python v2/aggregate_final.py --parent . --v2 v2 --out runs/report_final \
  --compare runs/compare/compare.json 2>&1 | tee -a "$LOG" || echo "[finalize] aggregate FAILED" | tee -a "$LOG"

# 2) in-distribution videos (all 10 tasks, flow|v1|v2)
python make_videos_indist.py --out runs/videos 2>&1 | grep -E "vid\]" | tee -a "$LOG" \
  || echo "[finalize] indist videos FAILED" | tee -a "$LOG"

# 3) perturbed videos (best-effort)
python make_videos_pert.py --out runs/videos 2>&1 | grep -E "vidp\]" | tee -a "$LOG" \
  || echo "[finalize] pert videos FAILED" | tee -a "$LOG"

# 4) assemble the deliverable folder + zip
D=DeepONet_Results
rm -rf "$D" "$D.zip"; mkdir -p "$D/plots" "$D/videos" "$D/idea" "$D/data"
cp runs/report_final/*.png "$D/plots/" 2>/dev/null
cp runs/report/*.png "$D/plots/" 2>/dev/null            # v1 3-seed plots
cp runs/report_final/REPORT.md "$D/" 2>/dev/null
cp runs/videos/*.mp4 "$D/videos/" 2>/dev/null
cp runs/videos/*.png "$D/videos/" 2>/dev/null
cp v2/IDEA_v2.md PORTING_TO_PI0.md "$D/idea/" 2>/dev/null
cp runs/report_final/report.json runs/compare/compare.json "$D/data/" 2>/dev/null
cp runs/eval_indist/success_rates.json "$D/data/indist_v1flow.json" 2>/dev/null
cp runs/eval_plus/robustness_plus.json "$D/data/robustness_v1flow.json" 2>/dev/null
cp v2/runs/eval_v2_indist/success_rates.json "$D/data/indist_v2.json" 2>/dev/null
cp v2/runs/eval_v2_plus/robustness_plus.json "$D/data/robustness_v2.json" 2>/dev/null
zip -rq "$D.zip" "$D" 2>/dev/null

echo "[finalize] DONE $(date) -> $D/ and $D.zip" | tee -a "$LOG"
ls -R "$D" | tee -a "$LOG"
touch ALL_DONE
