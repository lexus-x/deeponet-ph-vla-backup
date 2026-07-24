#!/usr/bin/env python
"""
sweep_report.py
===============
Collect the lambda-sweep results (LIBERO-Spatial) and plot in-distribution
accuracy and robustness vs lambda, with the baseline (lambda=0) as a reference
line and lambda=0.1 from the main strong run included.

Reads:
  output/results_strong/success_rates.json   (baseline @ lambda=0, ph @ lambda=0.1)
  output/results_sweep/lambda_*.json         (ph @ other lambdas, --skip_baseline)
Writes:
  output/plots_strong/lambda_sweep_spatial.{png,pdf}
  output/results_sweep/sweep_summary.json
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRONG = Path("output/results_strong/success_rates.json")
SWEEP_DIR = Path("output/results_sweep")
OUTPLOT = Path("output/plots_strong")
IND, PERT = "LIBERO-SPATIAL", "LIBERO-SPATIAL-V"


def ph_points():
    pts = {}  # lambda -> (indist%, robust%)
    # lambda=0.1 from the strong run
    if STRONG.exists():
        d = json.loads(STRONG.read_text())
        a = d.get(IND, {}).get("ph", {}).get("average")
        r = d.get(PERT, {}).get("ph", {}).get("average_over_perturbations")
        if a is not None:
            pts[0.1] = (a * 100, (r or 0) * 100)
    # other lambdas from the sweep
    for f in sorted(glob.glob(str(SWEEP_DIR / "lambda_*.json"))):
        lam = float(re.search(r"lambda_([0-9p.]+)\.json", f).group(1).replace("p", "."))
        d = json.loads(Path(f).read_text())
        a = d.get(IND, {}).get("ph", {}).get("average")
        r = d.get(PERT, {}).get("ph", {}).get("average_over_perturbations")
        if a is not None:
            pts[lam] = (a * 100, (r or 0) * 100)
    return dict(sorted(pts.items()))


def baseline_refs():
    if not STRONG.exists():
        return None, None
    d = json.loads(STRONG.read_text())
    b_ind = d.get(IND, {}).get("baseline", {}).get("average")
    b_rob = d.get(PERT, {}).get("baseline", {}).get("average_over_perturbations")
    return (b_ind * 100 if b_ind is not None else None,
            b_rob * 100 if b_rob is not None else None)


def main():
    pts = ph_points()
    b_ind, b_rob = baseline_refs()
    if not pts:
        print("[sweep] no PH points found"); return
    lams = list(pts)
    acc = [pts[l][0] for l in lams]
    rob = [pts[l][1] for l in lams]

    OUTPLOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(lams, acc, "o-", color="#1f77b4", label="flow+PH: in-distribution acc")
    ax.plot(lams, rob, "s-", color="#d62728", label="flow+PH: robustness (avg perturb)")
    if b_ind is not None:
        ax.axhline(b_ind, ls="--", color="#1f77b4", alpha=0.6, label=f"baseline acc ({b_ind:.1f}%)")
    if b_rob is not None:
        ax.axhline(b_rob, ls="--", color="#d62728", alpha=0.6, label=f"baseline robustness ({b_rob:.1f}%)")
    ax.set_xscale("log"); ax.set_xlabel("lambda_ph (log scale)")
    ax.set_ylabel("success rate (%)"); ax.set_ylim(0, 100)
    ax.set_title("LIBERO-Spatial: PH lambda sweep vs baseline\n(points above dashed lines = PH beats baseline)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for l, a, r in zip(lams, acc, rob):
        ax.annotate(f"{a:.0f}", (l, a), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
        ax.annotate(f"{r:.0f}", (l, r), textcoords="offset points", xytext=(0, -12), fontsize=7, ha="center")
    for ext in ("png", "pdf"):
        fig.savefig(OUTPLOT / f"lambda_sweep_spatial.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    summary = {"baseline": {"indist": b_ind, "robustness": b_rob},
               "ph_by_lambda": {str(l): {"indist": pts[l][0], "robustness": pts[l][1]} for l in lams}}
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print("[sweep] baseline  indist=%.1f robust=%.1f" % (b_ind or 0, b_rob or 0))
    for l in lams:
        print(f"[sweep] lambda={l:<5} indist={pts[l][0]:.1f}  robust={pts[l][1]:.1f}")
    print(f"[sweep] wrote {OUTPLOT}/lambda_sweep_spatial.png + sweep_summary.json")


if __name__ == "__main__":
    main()
