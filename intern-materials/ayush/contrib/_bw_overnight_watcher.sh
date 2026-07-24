#!/usr/bin/env bash
# Overnight watcher: every 15m refresh SUMMARY on blackwell + attempt scp note for local.
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test/DeepONet PH/v2"
OUT_REMOTE="$ROOT/plus_multisuite_campaign/OVERNIGHT_SUMMARY.md"
OUT_JSON="$ROOT/plus_multisuite_campaign/OVERNIGHT_SUMMARY.json"
EXEC="$ROOT/exec_campaign_results"
POD_LOG="$ROOT/pod_train_spatial_30k/train.log"
POD_CKPT="$ROOT/pod_train_spatial_30k/checkpoints"
PLUS="$ROOT/plus_multisuite_campaign"
PY="/home/user/Desktop/Ayush PH test/venv/bin/python"

aggregate_once () {
"$PY" - <<'PY'
import json, glob, os, re, time
from pathlib import Path
from datetime import datetime, timezone

root = Path("/home/user/Desktop/Ayush PH test/DeepONet PH/v2")
plus = root / "plus_multisuite_campaign"
exec_dir = root / "exec_campaign_results"
pod_log = root / "pod_train_spatial_30k" / "train.log"
pod_ckpt = root / "pod_train_spatial_30k" / "checkpoints"

summary = {
    "updated": datetime.now().astimezone().isoformat(timespec="seconds"),
    "plus": {},
    "pod30k": {},
    "long_boundary": {},
    "verdict": "",
}

# --- Plus ---
for p in sorted(plus.glob("*/robustness_plus.json")):
    tag = p.parent.name
    if tag.endswith("_aborted") or tag.endswith("_n12_aborted"):
        continue
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        summary["plus"][tag] = {"error": str(e)}
        continue
    # find model key
    models = [k for k in d if not k.startswith("_")]
    entry = {"path": str(p), "models": {}}
    for m in models:
        md = d[m]
        if not isinstance(md, dict):
            continue
        cats = {}
        for c, cv in md.items():
            if c in ("robustness_average",) or not isinstance(cv, dict):
                continue
            if "average" in cv:
                cats[c] = cv["average"]
        entry["models"][m] = {
            "robustness_average": md.get("robustness_average"),
            "per_category": cats,
            "n_tasks_done": sum(len(cv.get("per_task", {})) for cv in md.values() if isinstance(cv, dict) and "per_task" in cv),
        }
    entry["config"] = d.get("_config", {})
    summary["plus"][tag] = entry

# --- POD ---
pod = {"log_exists": pod_log.exists(), "latest_ckpt": None, "last_lines": []}
if pod_ckpt.exists():
    latest = pod_ckpt / "LATEST.txt"
    if latest.exists():
        pod["latest_ckpt"] = latest.read_text().strip()
    else:
        nums = sorted([x.name for x in pod_ckpt.iterdir() if x.name.isdigit()], key=lambda s: int(s))
        pod["latest_ckpt"] = nums[-1] if nums else None
if pod_log.exists():
    lines = pod_log.read_text(errors="replace").splitlines()
    pod["last_lines"] = lines[-8:]
    for line in reversed(lines):
        m = re.search(r"step\s+(\d+)", line)
        if m:
            pod["last_step"] = int(m.group(1))
            break
    if any("DONE" in L for L in lines[-5:]):
        pod["status"] = "DONE"
    elif any("stage2" in L for L in lines):
        pod["status"] = "stage2"
    elif any("stage1" in L for L in lines):
        pod["status"] = "stage1"
    else:
        pod["status"] = "starting"
summary["pod30k"] = pod

# --- Long boundary / exec campaign (offline) ---
boundary = {"ens": None, "pin": None, "note": "from exec_campaign_results SUMMARY"}
sum_path = exec_dir / "SUMMARY.json"
if sum_path.exists():
    try:
        es = json.loads(sum_path.read_text())
        boundary["ens"] = es.get("ens")
        boundary["pin"] = es.get("pin")
        boundary["anchors"] = es.get("_anchors")
        # Mechanism narrative
        ens = es.get("ens") or {}
        per = ens.get("per_task") or {}
        if per:
            lows = sorted(per.items(), key=lambda kv: kv[1])[:3]
            highs = sorted(per.items(), key=lambda kv: -kv[1])[:3]
            boundary["ens_worst_tasks"] = lows
            boundary["ens_best_tasks"] = highs
            boundary["mechanism"] = (
                f"Long ens mean={ens.get('mean_sr')}% vs m3_r5={es.get('_anchors',{}).get('m3_r5')}%; "
                f"pin status={((es.get('pin') or {}).get('status'))}. "
                "Averaging/pinning across mode jumps hurts closed-loop SR (BID-consistent)."
            )
    except Exception as e:
        boundary["error"] = str(e)
summary["long_boundary"] = boundary

# --- Verdict ---
plus_avgs = []
for tag, ent in summary["plus"].items():
    for m, md in (ent.get("models") or {}).items():
        if md.get("robustness_average") is not None:
            plus_avgs.append((tag, m, md["robustness_average"]))
if plus_avgs:
    # compare m3 vs flow per suite if both present
    by_suite = {}
    for tag, m, avg in plus_avgs:
        suite = tag.split("_", 1)[-1] if "_" in tag else tag
        by_suite.setdefault(suite, {})[m] = avg
    wins = []
    for suite, mm in by_suite.items():
        if "m3" in mm and "flow" in mm:
            delta = mm["m3"] - mm["flow"]
            wins.append(f"{suite}: m3={mm['m3']:.3f} flow={mm['flow']:.3f} d={delta:+.3f}")
    if wins:
        summary["verdict"] = "Plus partial/complete — " + "; ".join(wins)
    else:
        summary["verdict"] = "Plus in progress — " + ", ".join(f"{t}/{m}={a:.3f}" for t,m,a in plus_avgs)
else:
    summary["verdict"] = "Plus still warming up; POD=" + str(pod.get("status")) + " step=" + str(pod.get("last_step"))

# write JSON + MD
out_json = plus / "OVERNIGHT_SUMMARY.json"
out_md = plus / "OVERNIGHT_SUMMARY.md"
out_json.write_text(json.dumps(summary, indent=2))

lines = [
    f"# DeepONet overnight SUMMARY",
    f"",
    f"**Updated:** {summary['updated']}",
    f"",
    f"**Verdict:** {summary['verdict']}",
    f"",
    f"## LIBERO-Plus",
    f"",
]
for tag, ent in sorted(summary["plus"].items()):
    lines.append(f"### `{tag}`")
    for m, md in (ent.get("models") or {}).items():
        lines.append(f"- **{m}** robustness_avg=`{md.get('robustness_average')}` tasks_done=`{md.get('n_tasks_done')}`")
        if md.get("per_category"):
            for c, a in md["per_category"].items():
                lines.append(f"  - {c}: {a}")
    lines.append("")
lines += [
    f"## POD-30K train",
    f"",
    f"- status: `{pod.get('status')}`",
    f"- last_step: `{pod.get('last_step')}`",
    f"- latest_ckpt: `{pod.get('latest_ckpt')}`",
    f"",
    f"```",
    *pod.get("last_lines", [])[-6:],
    f"```",
    f"",
    f"## Long boundary / exec (offline)",
    f"",
    f"- {boundary.get('mechanism', 'n/a')}",
    f"- ens: `{json.dumps(boundary.get('ens'))}`",
    f"- pin: `{json.dumps(boundary.get('pin'))}`",
    f"",
    f"## Wake-up checklist",
    f"",
    f"1. If Object/Long Plus m3 >> flow → draft co-lead Plus table",
    f"2. Else use POD mid-ckpt + mechanism narrative",
    f"3. a100 was left alone (bridge pretrain)",
    f"",
]
out_md.write_text("\n".join(lines))
print(out_md)
print(summary["verdict"])
PY
}

# one-shot boundary already inside aggregate
echo "WATCHER_START $(date)"
while true; do
  aggregate_once || echo "aggregate failed $(date)"
  # also mirror to home for easy scp
  cp -f "$OUT_REMOTE" "$HOME/deeponet_OVERNIGHT_SUMMARY.md" 2>/dev/null || true
  cp -f "$OUT_JSON" "$HOME/deeponet_OVERNIGHT_SUMMARY.json" 2>/dev/null || true
  sleep 900
done
