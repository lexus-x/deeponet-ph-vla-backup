#!/usr/bin/env python
"""Save a suite's flow eval two ways: (1) the FULL all-task average (the official
LIBERO number) and (2) a SUPPLEMENTARY average excluding one task (clearly labeled).
Writes summary_full.json, summary_excl_task<N>.json, summary_both.csv into the eval dir."""
import json, argparse, csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--suite_dir", required=True)
ap.add_argument("--suite", required=True)              # libero_spatial / libero_object / libero_10
ap.add_argument("--model", default="flow")
ap.add_argument("--eval_subdir", default="eval_flow")
ap.add_argument("--drop_task", type=int, default=None, help="task id to exclude (supplementary avg)")
ap.add_argument("--drop", default=None, help="'auto' = exclude the single lowest-scoring task")
a = ap.parse_args()

SD = Path(a.suite_dir)
bench = a.suite.replace("libero_", "LIBERO-").upper()
out = SD / "runs" / a.eval_subdir
d = json.loads((out / "success_rates.json").read_text())
g = d[bench][a.model]
pt = {int(t): v["success_rate"] * 100 for t, v in g["per_task"].items()}
tasks = sorted(pt)
full = sum(pt.values()) / len(pt)

drop = a.drop_task
if a.drop == "auto":
    drop = min(pt, key=pt.get)

# (1) full average — the official number
(out / "summary_full.json").write_text(json.dumps({
    "suite": bench, "model": a.model, "n_tasks": len(pt),
    "average_all_tasks_pct": round(full, 2),
    "per_task_pct": {f"task{t}": pt[t] for t in tasks},
    "note": "OFFICIAL LIBERO number = average over ALL tasks.",
}, indent=2))
print(f"[{bench}] FULL {len(pt)}-task average = {full:.1f}%")

rows = [(f"full_{len(pt)}task_avg", f"{full:.1f}")]
if drop is not None:
    kept = {t: pt[t] for t in tasks if t != drop}
    ex = sum(kept.values()) / len(kept)
    (out / f"summary_excl_task{drop}.json").write_text(json.dumps({
        "suite": bench, "model": a.model,
        "excluded_task": f"task{drop}", "excluded_task_success_pct": pt.get(drop),
        "n_tasks_kept": len(kept), "average_excluding_task_pct": round(ex, 2),
        "per_task_kept_pct": {f"task{t}": kept[t] for t in sorted(kept)},
        "note": "SUPPLEMENTARY analysis only (hardest task removed). NOT the official LIBERO number.",
    }, indent=2))
    print(f"[{bench}] EXCL task{drop} ({pt.get(drop):.0f}%) -> {len(kept)}-task average = {ex:.1f}%")
    rows.append((f"excl_task{drop}_{len(kept)}task_avg", f"{ex:.1f}"))

with open(out / "summary_both.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "value_pct"]); w.writerows(rows)
print(f"[{bench}] wrote summary_full.json"
      + (f" + summary_excl_task{drop}.json" if drop is not None else "")
      + " + summary_both.csv ->", out)
