#!/usr/bin/env python
"""
aggregate.py
============
Aggregate the multi-seed evaluation into a defensible head-to-head report:

  * mean +/- std over seeds for in-distribution accuracy and LIBERO-Plus
    robustness (overall + per perturbation category) for M1/M3/M4;
  * paired significance tests (M3 vs M1, M4 vs M3, M4 vs M1) using per-unit
    pairing (per-task x seed for accuracy; per-category x seed for robustness)
    -- both a paired t-test and a Wilcoxon signed-rank (reported together);
  * the deterministic efficiency gate (params, latency) from compare.json;
  * bar charts with error bars + a summary table (CSV + JSON).

Runs on partial results too (skips missing models), so it can be used to peek
mid-eval.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as _st
except Exception:
    _st = None

MODELS = ["m1", "m3", "m4"]
LABELS = {"m1": "M1 flow-matching", "m3": "M3 DeepONet", "m4": "M4 DeepONet+PH"}
COLORS = {"m1": "#2E86DE", "m3": "#10AC84", "m4": "#EE5253"}
CATS = ["Camera Viewpoints", "Light Conditions", "Sensor Noise", "Background Textures",
        "Objects Layout", "Robot Initial States", "Language Instructions"]


def _parse(name):
    m = re.match(r"(m\d+)_s(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (None, None)


# --------------------------------------------------------------- load per-seed
def load_indist(path):
    """-> {model_type: {seed: {'avg': float, 'per_task': {tid: rate}}}}"""
    out = defaultdict(dict)
    if not Path(path).exists():
        return out
    d = json.loads(Path(path).read_text())
    bench = d.get("LIBERO-SPATIAL", {})
    for name, rec in bench.items():
        mt, seed = _parse(name)
        if mt is None or "per_task" not in rec:
            continue
        per_task = {k: v["success_rate"] for k, v in rec["per_task"].items()}
        out[mt][seed] = {"avg": rec.get("average"), "per_task": per_task}
    return out


def load_plus(path):
    """-> {model_type: {seed: {'avg': float, 'per_cat': {cat: rate}}}}"""
    out = defaultdict(dict)
    if not Path(path).exists():
        return out
    d = json.loads(Path(path).read_text())
    for name, rec in d.items():
        if name.startswith("_"):
            continue
        mt, seed = _parse(name)
        if mt is None:
            continue
        per_cat = {c: rec[c]["average"] for c in CATS
                   if isinstance(rec.get(c), dict) and rec[c].get("average") is not None}
        out[mt][seed] = {"avg": rec.get("robustness_average"), "per_cat": per_cat}
    return out


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


# --------------------------------------------------------------- paired tests
def paired_test(a, b):
    """Paired a-b: returns dict with mean diff, t-test p, wilcoxon p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"n": int(len(a)), "mean_diff": None, "t_p": None, "wilcoxon_p": None}
    res = {"n": int(len(a)), "mean_diff": float(np.mean(a - b))}
    if _st is not None:
        try:
            res["t_p"] = float(_st.ttest_rel(a, b).pvalue)
        except Exception:
            res["t_p"] = None
        try:
            res["wilcoxon_p"] = float(_st.wilcoxon(a, b).pvalue) if np.any(a - b) else 1.0
        except Exception:
            res["wilcoxon_p"] = None
    else:
        res["t_p"] = res["wilcoxon_p"] = None
    return res


def collect_pairs_indist(indist, ma, mb):
    """Pair per (task, seed) in-dist success rates of ma vs mb."""
    a, b = [], []
    seeds = set(indist.get(ma, {})) & set(indist.get(mb, {}))
    for s in sorted(seeds):
        pa, pb = indist[ma][s]["per_task"], indist[mb][s]["per_task"]
        for t in set(pa) & set(pb):
            a.append(pa[t]); b.append(pb[t])
    return a, b


def collect_pairs_plus(plus, ma, mb):
    """Pair per (category, seed) robustness of ma vs mb."""
    a, b = [], []
    seeds = set(plus.get(ma, {})) & set(plus.get(mb, {}))
    for s in sorted(seeds):
        ca, cb = plus[ma][s]["per_cat"], plus[mb][s]["per_cat"]
        for c in set(ca) & set(cb):
            a.append(ca[c]); b.append(cb[c])
    return a, b


# --------------------------------------------------------------- plots
def bar_with_err(ax, means, stds, title, ylabel):
    xs = np.arange(len(MODELS))
    ax.bar(xs, [100 * (means[m] or 0) for m in MODELS],
           yerr=[100 * (stds[m] or 0) for m in MODELS], capsize=6,
           color=[COLORS[m] for m in MODELS])
    for i, m in enumerate(MODELS):
        if means[m] is not None:
            ax.text(i, 100 * means[m] + 1.5, f"{100*means[m]:.1f}", ha="center",
                    fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indist", required=True)
    ap.add_argument("--plus", required=True)
    ap.add_argument("--compare", default="runs/compare/compare.json")
    ap.add_argument("--out", default="runs/report")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    indist = load_indist(args.indist)
    plus = load_plus(args.plus)

    report = {"accuracy": {}, "robustness": {}, "robustness_per_cat": {}, "significance": {}}

    # accuracy (in-dist) mean+-std over seeds
    acc_m, acc_s = {}, {}
    for mt in MODELS:
        mu, sd, n = mean_std([indist.get(mt, {}).get(s, {}).get("avg")
                              for s in indist.get(mt, {})])
        acc_m[mt], acc_s[mt] = mu, sd
        report["accuracy"][mt] = {"mean": mu, "std": sd, "n_seeds": n}

    # robustness overall mean+-std
    rob_m, rob_s = {}, {}
    for mt in MODELS:
        mu, sd, n = mean_std([plus.get(mt, {}).get(s, {}).get("avg") for s in plus.get(mt, {})])
        rob_m[mt], rob_s[mt] = mu, sd
        report["robustness"][mt] = {"mean": mu, "std": sd, "n_seeds": n}

    # robustness per category mean+-std
    for c in CATS:
        report["robustness_per_cat"][c] = {}
        for mt in MODELS:
            mu, sd, n = mean_std([plus.get(mt, {}).get(s, {}).get("per_cat", {}).get(c)
                                  for s in plus.get(mt, {})])
            report["robustness_per_cat"][c][mt] = {"mean": mu, "std": sd}

    # significance (paired)
    for (ma, mb) in [("m3", "m1"), ("m4", "m3"), ("m4", "m1")]:
        a, b = collect_pairs_indist(indist, ma, mb)
        report["significance"][f"acc_{ma}_vs_{mb}"] = paired_test(a, b)
        a, b = collect_pairs_plus(plus, ma, mb)
        report["significance"][f"rob_{ma}_vs_{mb}"] = paired_test(a, b)

    # efficiency gate
    if Path(args.compare).exists():
        report["efficiency_gate"] = json.loads(Path(args.compare).read_text())

    (out / "report.json").write_text(json.dumps(report, indent=2))

    # ---- plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bar_with_err(axes[0], acc_m, acc_s, "In-distribution accuracy (LIBERO-Spatial)\nmean +/- std over seeds", "success (%)")
    bar_with_err(axes[1], rob_m, rob_s, "Robustness (LIBERO-Plus, 7 dims)\nmean +/- std over seeds", "success (%)")
    fig.tight_layout(); fig.savefig(out / "accuracy_robustness.png", dpi=150); plt.close(fig)

    # per-category robustness grouped
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(CATS)); w = 0.26
    for i, mt in enumerate(MODELS):
        means = [100 * (report["robustness_per_cat"][c][mt]["mean"] or 0) for c in CATS]
        errs = [100 * (report["robustness_per_cat"][c][mt]["std"] or 0) for c in CATS]
        ax.bar(x + (i - 1) * w, means, w, yerr=errs, capsize=3, label=LABELS[mt], color=COLORS[mt])
    ax.set_xticks(x); ax.set_xticklabels([c.replace(" ", "\n") for c in CATS], fontsize=8)
    ax.set_ylim(0, 100); ax.set_ylabel("success (%)"); ax.legend()
    ax.set_title("LIBERO-Plus robustness per perturbation dimension (mean +/- std)", fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(out / "robustness_per_category.png", dpi=150); plt.close(fig)

    # per-task in-dist (mean over seeds)
    tids = sorted({t for mt in MODELS for s in indist.get(mt, {})
                   for t in indist[mt][s]["per_task"]}, key=int)
    if tids:
        fig, ax = plt.subplots(figsize=(13, 5.5))
        x = np.arange(len(tids)); w = 0.26
        for i, mt in enumerate(MODELS):
            means = []
            for t in tids:
                vals = [indist[mt][s]["per_task"].get(t) for s in indist.get(mt, {})
                        if t in indist[mt][s]["per_task"]]
                means.append(100 * np.mean(vals) if vals else 0)
            ax.bar(x + (i - 1) * w, means, w, label=LABELS[mt], color=COLORS[mt])
        ax.set_xticks(x); ax.set_xticklabels([f"task{t}" for t in tids], fontsize=9, rotation=30)
        ax.set_ylim(0, 100); ax.set_ylabel("success (%)"); ax.legend()
        ax.set_title("In-distribution per-task success (mean over seeds)", fontweight="bold")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout(); fig.savefig(out / "per_task.png", dpi=150); plt.close(fig)

    # ---- printed summary
    def fmt(mu, sd): return f"{100*mu:.1f}+/-{100*sd:.1f}" if mu is not None else "  --  "
    print("\n================= HEAD-TO-HEAD (mean +/- std over seeds) =================")
    print(f"{'model':18s} {'in-dist acc':>14s} {'robustness':>14s}")
    for mt in MODELS:
        print(f"{LABELS[mt]:18s} {fmt(acc_m[mt], acc_s[mt] or 0):>14s} {fmt(rob_m[mt], rob_s[mt] or 0):>14s}")
    print("\n--- paired significance (positive mean_diff favours the first model) ---")
    for k, v in report["significance"].items():
        if v["mean_diff"] is not None:
            print(f"{k:16s} n={v['n']:3d} diff={100*v['mean_diff']:+5.1f}pp "
                  f"t_p={v['t_p']:.3f} wilcoxon_p={v['wilcoxon_p']}")
    if "efficiency_gate" in report:
        g = report["efficiency_gate"]
        print(f"\n--- efficiency gate (deterministic) ---")
        print(f"head params: flow {g['flow']['params']['head']/1e6:.1f}M vs "
              f"deeponet {g['deeponet']['params']['head']/1e6:.2f}M "
              f"({g['_comparison']['head_param_ratio_flow_over_deeponet']:.0f}x)")
        print(f"latency: flow {g['flow']['latency_ms_mean']:.0f}ms vs "
              f"deeponet {g['deeponet']['latency_ms_mean']:.0f}ms "
              f"({g['_comparison']['latency_speedup_flow_over_deeponet']:.1f}x faster)")
    print(f"\nsaved -> {out}/report.json + plots")


if __name__ == "__main__":
    main()
