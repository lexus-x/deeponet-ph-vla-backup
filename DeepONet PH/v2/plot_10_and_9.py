#!/usr/bin/env python
"""Make TWO per-task bar plots for a suite's flow eval:
   (1) all 10 tasks  + 10-task average annotated
   (2) 9 tasks (one excluded) + 9-task average annotated
Also (re)saves the two averages to summary_full.json / summary_excl_task<N>.json / summary_both.csv.
Usage:
  python plot_10_and_9.py --suite_dir ../paper_repro/Spatial --suite libero_spatial --drop_task 5 --label "Spatial flow 30K"
  python plot_10_and_9.py --suite_dir ../paper_repro/Object  --suite libero_object  --drop auto --label "Object flow 30K"
"""
import json, csv, argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--suite_dir", required=True)
ap.add_argument("--suite", required=True)
ap.add_argument("--model", default="flow")
ap.add_argument("--eval_subdir", default="eval_flow")
ap.add_argument("--drop_task", type=int, default=None)
ap.add_argument("--drop", default=None, help="'auto' = drop the single lowest task")
ap.add_argument("--label", default=None)
a = ap.parse_args()

SD = Path(a.suite_dir)
bench = a.suite.replace("libero_", "LIBERO-").upper()
label = a.label or bench
outdir = SD / "plots"; outdir.mkdir(parents=True, exist_ok=True)
evd = SD / "runs" / a.eval_subdir
g = json.loads((evd / "success_rates.json").read_text())[bench][a.model]
pt = {int(t): v["success_rate"] * 100 for t, v in g["per_task"].items()}
tasks = sorted(pt)
full = sum(pt.values()) / len(pt)

drop = a.drop_task
if a.drop == "auto":
    drop = min(pt, key=pt.get)

def bar(ts, title, fname, avg):
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [pt[t] for t in ts]
    bars = ax.bar([f"task{t}" for t in ts], vals, color="#2E86DE")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=9)
    ax.axhline(avg, ls="--", color="#E74C3C", lw=2)
    ax.text(len(ts) - 0.5, avg + 2, f"avg = {avg:.1f}%", color="#E74C3C", ha="right", fontweight="bold")
    ax.set_ylim(0, 108); ax.set_ylabel("success rate (%)")
    ax.set_title(title, fontweight="bold"); ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(outdir / fname, dpi=150); plt.close(fig)
    print("  saved", fname)

# (1) all 10 tasks
bar(tasks, f"{label} — all {len(tasks)} tasks  (avg {full:.1f}%)", f"{a.suite}_all{len(tasks)}tasks.png", full)
# (2) excluding one task
if drop is not None:
    kept = [t for t in tasks if t != drop]
    ex = sum(pt[t] for t in kept) / len(kept)
    bar(kept, f"{label} — {len(kept)} tasks (excl. task{drop})  (avg {ex:.1f}%)",
        f"{a.suite}_excl_task{drop}_{len(kept)}tasks.png", ex)

# (re)save the numbers separately
(evd / "summary_full.json").write_text(json.dumps(
    {"suite": bench, "n_tasks": len(pt), "average_all_tasks_pct": round(full, 2),
     "per_task_pct": {f"task{t}": pt[t] for t in tasks},
     "note": "OFFICIAL LIBERO number = average over ALL tasks."}, indent=2))
rows = [(f"full_{len(pt)}task_avg", f"{full:.1f}")]
print(f"[{label}] FULL {len(pt)}-task avg = {full:.1f}%")
if drop is not None:
    (evd / f"summary_excl_task{drop}.json").write_text(json.dumps(
        {"suite": bench, "excluded_task": f"task{drop}", "excluded_task_success_pct": pt.get(drop),
         "n_tasks_kept": len(kept), "average_excluding_task_pct": round(ex, 2),
         "per_task_kept_pct": {f"task{t}": pt[t] for t in kept},
         "note": "SUPPLEMENTARY (hardest task removed). NOT the official LIBERO number."}, indent=2))
    rows.append((f"excl_task{drop}_{len(kept)}task_avg", f"{ex:.1f}"))
    print(f"[{label}] EXCL task{drop} -> {len(kept)}-task avg = {ex:.1f}%")
with open(evd / "summary_both.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "value_pct"]); w.writerows(rows)
print(f"[{label}] plots -> {outdir} | summaries -> {evd}")
