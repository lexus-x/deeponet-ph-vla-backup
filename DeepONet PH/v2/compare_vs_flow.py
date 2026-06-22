#!/usr/bin/env python
"""
compare_vs_flow.py
==================
Per-suite comparison of DeepONet (m3) and DeepONet+PH (m4) vs the SmolVLA
flow-matching baseline (30K, batch 48). For one suite it produces:
  - <suite>_compare_vs_flow.png : grouped per-task bars (flow / m3 / m4) + avg lines
  - <suite>_compare_vs_flow.csv : per-task table + averages + delta-vs-flow + BEAT?
and prints the table + whether m3/m4 beat the flow average.

Flow numbers come from ../paper_repro/<Suite>/runs/eval_flow/.
m3/m4 come from ./deeponet_results/<Suite>/runs/eval_{m3,m4}/.
Run from inside DeepONet PH/v2/.
"""
import json, csv, argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SUITE_KEY = {"Spatial": "LIBERO-SPATIAL", "Object": "LIBERO-OBJECT", "Long": "LIBERO-10"}

def load_pertask(eval_dir: Path, suite_key: str):
    """Return {task_idx:int -> pct:float} and the average, from summary_full or success_rates."""
    sf = eval_dir / "summary_full.json"
    if sf.exists():
        d = json.loads(sf.read_text())
        pt = {int(k.replace("task", "")): float(v) for k, v in d["per_task_pct"].items()}
        return pt, float(d["average_all_tasks_pct"])
    sr = eval_dir / "success_rates.json"
    if sr.exists():
        d = json.loads(sr.read_text())
        g = d[suite_key]
        model = next(m for m in g if not m.startswith("_"))
        pt = {int(t): v["success_rate"] * 100 for t, v in g[model]["per_task"].items()}
        return pt, sum(pt.values()) / len(pt)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITE_KEY))
    ap.add_argument("--flow_root", default="../paper_repro")
    ap.add_argument("--deeponet_root", default="./deeponet_results")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    sk = SUITE_KEY[a.suite]

    sources = {
        "flow (SmolVLA)": Path(a.flow_root) / a.suite / "runs" / "eval_flow",
        "m3 DeepONet":    Path(a.deeponet_root) / a.suite / "runs" / "eval_m3",
        "m4 DeepONet+PH": Path(a.deeponet_root) / a.suite / "runs" / "eval_m4",
    }
    data = {}
    for name, d in sources.items():
        pt, avg = load_pertask(d, sk)
        if pt is not None:
            data[name] = (pt, avg)
        else:
            print(f"  (skip {name}: no results yet at {d})")
    if "flow (SmolVLA)" not in data:
        raise SystemExit("flow baseline not found — cannot compare.")

    tasks = sorted(next(iter(data.values()))[0].keys())
    out_dir = Path(a.out_dir) if a.out_dir else Path(a.deeponet_root) / a.suite / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- plot ----
    colors = {"flow (SmolVLA)": "#888888", "m3 DeepONet": "#1f77b4", "m4 DeepONet+PH": "#d62728"}
    n = len(data); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 1.1), 5.5))
    for i, (name, (pt, avg)) in enumerate(data.items()):
        xs = [t + (i - (n - 1) / 2) * w for t in tasks]
        ax.bar(xs, [pt[t] for t in tasks], width=w, label=f"{name} (avg {avg:.1f}%)",
               color=colors.get(name))
        ax.axhline(avg, color=colors.get(name), ls="--", lw=1, alpha=0.7)
    ax.set_xticks(tasks); ax.set_xticklabels([f"t{t}" for t in tasks])
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"LIBERO-{a.suite}: DeepONet vs DeepONet+PH vs SmolVLA-flow (30K, batch 48)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png = out_dir / f"{a.suite}_compare_vs_flow.png"
    fig.savefig(png, dpi=130); plt.close(fig)

    # ---- csv + verdict ----
    flow_pt, flow_avg = data["flow (SmolVLA)"]
    csv_path = out_dir / f"{a.suite}_compare_vs_flow.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["task"] + list(data.keys()))
        for t in tasks:
            wri.writerow([f"task{t}"] + [f"{data[m][0][t]:.1f}" for m in data])
        wri.writerow(["AVERAGE"] + [f"{data[m][1]:.2f}" for m in data])

    print(f"\n===== LIBERO-{a.suite} — averages =====")
    for name, (pt, avg) in data.items():
        delta = avg - flow_avg
        verdict = "" if name.startswith("flow") else f"  ({'BEATS' if delta > 0 else 'below'} flow by {delta:+.2f} pt)"
        print(f"  {name:18s}: {avg:6.2f}%{verdict}")
    print(f"  plot -> {png}\n  csv  -> {csv_path}")

if __name__ == "__main__":
    main()
