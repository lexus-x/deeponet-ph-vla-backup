#!/usr/bin/env python
"""
per_task_winloss.py
===================
Per-task baseline-vs-PH comparison for every benchmark in the results JSON,
with a PH-minus-baseline delta panel (green = PH better, red = baseline better).
Makes the task-level structure explicit: PH wins big on some tasks, loses on
others, while the *average* is what the headline plots report.

Usage:
  python per_task_winloss.py --results output/results_strong/success_rates.json \
      --out output/plots_strong
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CB, CP = "#1f77b4", "#d62728"


def per_task_rates(node):
    pt = node["per_task"]
    tasks = sorted(pt, key=int)
    return tasks, np.array([pt[t]["success_rate"] * 100 for t in tasks])


def plot_bench(bench, base, ph, outdir):
    tasks, b = per_task_rates(base)
    _, p = per_task_rates(ph)
    delta = p - b
    avg_b, avg_p = b.mean(), p.mean()
    x = np.arange(len(tasks)); w = 0.4

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                   gridspec_kw={"height_ratios": [2, 1]})
    # top: grouped bars
    ax1.bar(x - w/2, b, w, label=f"baseline (avg {avg_b:.1f}%)", color=CB)
    ax1.bar(x + w/2, p, w, label=f"flow+PH (avg {avg_p:.1f}%)", color=CP)
    ax1.axhline(avg_b, color=CB, ls="--", lw=1, alpha=0.6)
    ax1.axhline(avg_p, color=CP, ls="--", lw=1, alpha=0.6)
    ax1.set_ylabel("success rate (%)"); ax1.set_ylim(0, 105)
    ax1.set_title(f"{bench}: per-task success — baseline vs flow+PH "
                  f"(PH avg {avg_p:.1f}% vs baseline {avg_b:.1f}%)")
    ax1.set_xticks(x); ax1.set_xticklabels([f"t{t}" for t in tasks])
    ax1.grid(alpha=0.3, axis="y"); ax1.legend()

    # bottom: delta
    colors = ["#2ca02c" if d > 0 else ("#d62728" if d < 0 else "#999999") for d in delta]
    ax2.bar(x, delta, color=colors)
    ax2.axhline(0, color="k", lw=0.8)
    for i, d in enumerate(delta):
        if abs(d) >= 0.1:
            ax2.text(i, d + (2 if d >= 0 else -4), f"{d:+.0f}", ha="center", fontsize=8)
    nwin = int((delta > 0).sum()); nlose = int((delta < 0).sum()); ntie = int((delta == 0).sum())
    ax2.set_ylabel("PH − baseline (pts)")
    ax2.set_title(f"PH effect per task   (PH better: {nwin}  |  worse: {nlose}  |  tie: {ntie}   "
                  f"net avg {delta.mean():+.1f} pts)")
    ax2.set_xticks(x); ax2.set_xticklabels([f"t{t}" for t in tasks])
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    safe = bench.replace("/", "_")
    for ext in ("png", "pdf"):
        fig.savefig(Path(outdir) / f"per_task_winloss_{safe}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[winloss] {bench}: PH better {nwin}, worse {nlose}, tie {ntie}; "
          f"net {delta.mean():+.1f} pts -> per_task_winloss_{safe}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="output/results_strong/success_rates.json")
    ap.add_argument("--out", default="output/plots_strong")
    args = ap.parse_args()
    d = json.loads(Path(args.results).read_text())
    for bench in d:
        if bench.startswith("_"):
            continue
        node = d[bench]
        if "baseline" in node and "ph" in node and "per_task" in node["baseline"]:
            plot_bench(bench, node["baseline"], node["ph"], args.out)
    print("[winloss] DONE")


if __name__ == "__main__":
    main()
