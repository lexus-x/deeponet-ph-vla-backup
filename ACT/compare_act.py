#!/usr/bin/env python
"""compare_act.py — per-suite comparison of ACT vs ACT+DeepONet vs ACT+DeepONet+PH.
Reads one in-dist success_rates.json (all 3 variants) + optional LIBERO-Plus json.
Writes <Suite>_compare.png (+csv) and prints the table."""
import json, csv, argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ORDER = ["act", "act_deeponet", "act_deeponet_ph"]
LBL = {"act": "ACT", "act_deeponet": "ACT+DeepONet", "act_deeponet_ph": "ACT+DeepONet+PH"}
COL = {"act": "#888888", "act_deeponet": "#1f77b4", "act_deeponet_ph": "#d62728"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)            # Spatial/Object/Long/Goal
    ap.add_argument("--indist", required=True)           # eval_indist/success_rates.json
    ap.add_argument("--plus", default=None)              # eval_plus/success_rates.json
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    sk = "LIBERO-" + ("10" if a.suite == "Long" else a.suite.upper())
    d = json.loads(Path(a.indist).read_text())[sk]
    models = [m for m in ORDER if m in d]
    tasks = sorted((int(t) for t in d[models[0]]["per_task"]), key=int)

    plus = {}
    if a.plus and Path(a.plus).exists():
        pj = json.loads(Path(a.plus).read_text())
        for m in models:
            cats = pj.get(m, {})
            avgs = [cats[c]["average"] for c in cats if isinstance(cats.get(c), dict) and cats[c].get("average") is not None]
            plus[m] = 100 * sum(avgs) / len(avgs) if avgs else None

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n = len(models); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 1.1), 5.5))
    for i, m in enumerate(models):
        pt = d[m]["per_task"]
        xs = [t + (i - (n - 1) / 2) * w for t in tasks]
        ys = [pt[str(t)]["success_rate"] * 100 for t in tasks]
        errs = [pt[str(t)].get("std", 0) * 100 for t in tasks]
        avg = d[m]["average"] * 100
        ax.bar(xs, ys, width=w, yerr=errs, capsize=2, color=COL[m],
               label=f"{LBL[m]} (avg {avg:.1f}%{', rob '+format(plus[m],'.1f')+'%' if plus.get(m) is not None else ''})")
        ax.axhline(avg, color=COL[m], ls="--", lw=1, alpha=0.6)
    ax.set_xticks(tasks); ax.set_xticklabels([f"t{t}" for t in tasks])
    ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"LIBERO-{a.suite}: ACT vs ACT+DeepONet vs ACT+DeepONet+PH (30K, batch 64, 3-seed)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"{a.suite}_compare.png", dpi=130); plt.close(fig)

    with open(out_dir / f"{a.suite}_compare.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["task"] + [LBL[m] for m in models])
        for t in tasks:
            wr.writerow([f"task{t}"] + [f"{d[m]['per_task'][str(t)]['success_rate']*100:.1f}" for m in models])
        wr.writerow(["AVG"] + [f"{d[m]['average']*100:.2f}" for m in models])
        wr.writerow(["ROBUST(LIBERO-Plus)"] + [f"{plus.get(m):.2f}" if plus.get(m) is not None else "-" for m in models])

    print(f"\n===== LIBERO-{a.suite} =====")
    for m in models:
        rob = f"   robustness {plus[m]:.1f}%" if plus.get(m) is not None else ""
        print(f"  {LBL[m]:18s}: {d[m]['average']*100:6.2f}% +/- {d[m].get('average_std',0)*100:.1f}{rob}")
    print(f"  plot -> {out_dir / (a.suite + '_compare.png')}")


if __name__ == "__main__":
    main()
