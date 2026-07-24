#!/usr/bin/env bash
# Parallel Long pinning/ensemble campaign (online HF cache under ~/.cache).
set -uo pipefail
BASE="$HOME/Desktop/Ayush PH test/DeepONet PH/v2"
CONTRIB="$HOME/Desktop/Ayush PH test/contrib_postjul15"
PY="$HOME/Desktop/Ayush PH test/venv/bin/python"
cd "$BASE" || exit 1

export HF_HOME=/home/user/.cache/huggingface
# Allow online; cache will fill after first download
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true
export MUJOCO_GL=osmesa TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export DEEPONET_P=256 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=16 DEEPONET_HEAD=deeponet
export PYTHONPATH="$CONTRIB:$BASE:${PYTHONPATH:-}"

OUT="$BASE/exec_campaign_results"
mkdir -p "$OUT"
LOG="$OUT/campaign.log"
: > "$LOG"
M3="$BASE/deeponet_results/Long/runs/m3_s0/checkpoints/30000"
NORM=policy_preprocessor_step_5_normalizer_processor.safetensors
DS=lerobot/libero_10_image
SUITE=libero_10

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "pinning campaign start on $(hostname)"

# Offline stats stub still needed (dataset metadata online check)
run_one () {
  local name=$1 mode=$2 tasks=$3
  local odir="$OUT/$name"
  mkdir -p "$odir"
  log "START $name mode=$mode tasks=$tasks"
  OFFLINE_STATS_SF="$M3/$NORM" "$PY" "$CONTRIB/eval_exec_offline.py" \
    --suite "$SUITE" --dataset "$DS" \
    --model "m3=deeponet=$M3" --exec "$mode" --replan 5 \
    --indist_episodes 20 --task_ids "$tasks" \
    --out "$odir" >>"$odir/run.log" 2>&1 \
    && log "DONE $name" || log "FAIL $name (see $odir/run.log)"
}

run_one m3_pin_t0-4    pin    "0,1,2,3,4"  &
run_one m3_pin_t5-9    pin    "5,6,7,8,9"  &
run_one m3_ens_t0-4    ens    "0,1,2,3,4"  &
run_one m3_ens_t5-9    ens    "5,6,7,8,9"  &
run_one m3_pinens_t0-4 pinens "0,1,2,3,4"  &
run_one m3_pinens_t5-9 pinens "5,6,7,8,9"  &
wait

"$PY" - "$OUT" <<'PY'
import json, glob, os, sys
out = sys.argv[1]
modes = {}
for f in sorted(glob.glob(os.path.join(out, "*", "success_rates.json"))):
    name = os.path.basename(os.path.dirname(f))
    d = json.load(open(f))
    for bench, bd in d.items():
        if bench.startswith("_"): continue
        for m, md in bd.items():
            if not isinstance(md, dict) or "per_task" not in md: continue
            mode = md.get("exec_mode")
            if not mode:
                # name like m3_pin_t0-4
                parts = name.split("_")
                mode = parts[1] if len(parts) > 1 else name
            modes.setdefault(mode, {}).update({k: v["success_rate"] for k, v in md["per_task"].items()})
summary = {}
for mode, tasks in modes.items():
    if not tasks: continue
    rates = list(tasks.values())
    summary[mode] = {
        "mean_sr": round(100 * sum(rates) / len(rates), 2),
        "n_tasks": len(rates),
        "per_task": {k: round(100 * v, 1) for k, v in sorted(tasks.items(), key=lambda x: int(x[0]))},
    }
summary["_anchors"] = {"m3_r5": 60.0, "flow_r5": 65.0, "flow_reported": 66.5}
path = os.path.join(out, "SUMMARY.json")
json.dump(summary, open(path, "w"), indent=2)
print("=== EXEC CAMPAIGN SUMMARY ===")
print(json.dumps(summary, indent=2))
PY

touch "$OUT/CAMPAIGN_DONE"
log "CAMPAIGN COMPLETE -> $OUT/SUMMARY.json"
