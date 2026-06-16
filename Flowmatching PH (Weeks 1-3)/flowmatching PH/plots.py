#!/usr/bin/env python
"""
plots.py
========
Generate result figures (PNG + PDF) and assemble summary.pdf. Generic over
benchmarks: it plots whatever in-distribution suites (e.g. LIBERO-10,
LIBERO_SPATIAL) and their "-V" perturbation counterparts are present in the
results JSON, and whatever training runs are passed in.

Usage
-----
  # default (v1 single-suite run):
  python plots.py

  # strong multi-suite run:
  python plots.py --results results_strong/success_rates.json \
      --compare results_strong/compare.json --out plots_strong \
      --runs "LIBERO-10:baseline=outputs/strong_libero10_baseline,LIBERO-10:ph=outputs/strong_libero10_ph,LIBERO_SPATIAL:baseline=outputs/strong_spatial_baseline,LIBERO_SPATIAL:ph=outputs/strong_spatial_ph"
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"baseline": "#1f77b4", "ph": "#d62728"}


def read_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def col(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, ValueError):
            out.append(np.nan)
    return np.array(out)


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


# --------------------------------------------------------------------------- training curves
def training_curves(runs, per, outdir):
    """runs: dict suite -> {variant -> dir}."""
    fname = "log_epoch.csv" if per == "epoch" else "log_step.csv"
    xkey = "epoch_global_step" if per == "epoch" else "global_step"
    for suite, variants in runs.items():
        data = {v: read_csv(Path(d) / fname) for v, d in variants.items()}
        if not any(data.values()):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, metric, title in zip(
                axes, ["flow_matching_loss", "l1_loss", "ph_loss"],
                ["Flow-matching loss", "L1 loss", "PH loss"]):
            for v, rows in data.items():
                if rows:
                    ax.plot(col(rows, xkey), col(rows, metric), label=v,
                            color=C.get(v, None), lw=1.3)
            ax.set_title(title); ax.set_xlabel(f"step ({per})")
            ax.grid(alpha=0.3); ax.legend()
        fig.suptitle(f"{suite} — training curves (per {per})")
        _save(fig, outdir, f"training_curves_{suite}_per_{per}")


# --------------------------------------------------------------------------- success bars
def indist_bars(sr, outdir):
    """Per-task success for each in-distribution benchmark (no '-V')."""
    for bench in sr:
        if bench.startswith("_") or bench.endswith("-V"):
            continue
        models = {m: sr[bench][m] for m in ("baseline", "ph") if m in sr[bench]}
        if not models:
            continue
        tasks = sorted({t for m in models.values() for t in m.get("per_task", {})}, key=int)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = np.arange(len(tasks)); w = 0.38
        for i, (m, off) in enumerate([("baseline", -w/2), ("ph", w/2)]):
            if m not in models:
                continue
            vals = [models[m]["per_task"].get(t, {}).get("success_rate", np.nan)*100 for t in tasks]
            ax.bar(x + off, vals, w, label=f"{m} (avg {(_avg(models[m]))*100:.1f}%)", color=C[m])
        ax.set_xticks(x); ax.set_xticklabels([f"t{t}" for t in tasks])
        ax.set_ylabel("success rate (%)"); ax.set_ylim(0, 100)
        ax.set_title(f"{bench} per-task success (in-distribution)")
        ax.grid(alpha=0.3, axis="y"); ax.legend()
        _save(fig, outdir, f"success_{bench}_per_task")


def _avg(model_node):
    a = model_node.get("average")
    return a if a is not None else float(np.nan)


def perturbation_bars(sr, outdir):
    """Per-perturbation success for each '-V' benchmark."""
    perts = ["viewpoint", "lighting", "sensor_noise"]
    for bench in sr:
        if not bench.endswith("-V"):
            continue
        models = {m: sr[bench][m] for m in ("baseline", "ph") if m in sr[bench]}
        if not models:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(perts)); w = 0.38
        for m, off in [("baseline", -w/2), ("ph", w/2)]:
            if m not in models:
                continue
            vals = [models[m].get(p, {}).get("average", np.nan)*100 for p in perts]
            bars = ax.bar(x + off, vals, w, label=m, color=C[m])
            for r in bars:
                h = r.get_height()
                if not np.isnan(h):
                    ax.text(r.get_x()+r.get_width()/2, h+1, f"{h:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(perts)
        ax.set_ylabel("success rate (%)"); ax.set_ylim(0, 100)
        ax.set_title(f"{bench}: success by perturbation type")
        ax.grid(alpha=0.3, axis="y"); ax.legend()
        _save(fig, outdir, f"success_{bench}_by_perturbation")


# --------------------------------------------------------------------------- params/latency
def params_latency(cmp, outdir):
    if not cmp:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    params = cmp.get("params", {}).get("baseline", {})
    labels = [k for k in params if k != "TOTAL"]
    ax1.barh(labels, [params[k]/1e6 for k in labels], color="#4c72b0")
    ax1.set_xlabel("parameters (millions)")
    ax1.set_title(f"Parameters by submodule (identical both; TOTAL={params.get('TOTAL',0)/1e6:.1f}M)")
    ax1.grid(alpha=0.3, axis="x")
    lat = cmp.get("latency", {})
    if lat:
        ms = [lat["baseline"]["mean_ms"], lat["ph"]["mean_ms"]]
        er = [lat["baseline"]["std_ms"], lat["ph"]["std_ms"]]
        ax2.bar(["baseline", "ph"], ms, yerr=er, capsize=6, color=[C["baseline"], C["ph"]])
        for i, v in enumerate(ms):
            ax2.text(i, v, f"{v:.1f}ms", ha="center", va="bottom")
        ax2.set_ylabel("latency (ms, batch=1)")
        ax2.set_title(f"Inference latency ({lat['baseline']['n']} passes)\nPH train-only -> parity")
        ax2.grid(alpha=0.3, axis="y")
    _save(fig, outdir, "params_latency")


# --------------------------------------------------------------------------- headline
def headline(sr, cmp, outdir):
    benches_indist = [b for b in sr if not b.startswith("_") and not b.endswith("-V")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # (1) in-dist averages per suite
    ax = axes[0, 0]
    xs = np.arange(len(benches_indist)); w = 0.38
    for m, off in [("baseline", -w/2), ("ph", w/2)]:
        vals = [(sr[b].get(m, {}).get("average") or 0)*100 for b in benches_indist]
        ax.bar(xs+off, vals, w, label=m, color=C[m])
    ax.set_xticks(xs); ax.set_xticklabels(benches_indist, fontsize=8)
    ax.set_ylim(0, 100); ax.set_title("In-distribution avg success"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    # (2) -V averages per suite
    ax = axes[0, 1]
    benches_v = [b for b in sr if b.endswith("-V")]
    xs = np.arange(len(benches_v));
    for m, off in [("baseline", -w/2), ("ph", w/2)]:
        vals = [(sr[b].get(m, {}).get("average_over_perturbations") or 0)*100 for b in benches_v]
        ax.bar(xs+off, vals, w, label=m, color=C[m])
    ax.set_xticks(xs); ax.set_xticklabels(benches_v, fontsize=8)
    ax.set_ylim(0, 100); ax.set_title("Robustness avg (over perturbations)"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    # (3) first -V bench by perturbation
    ax = axes[1, 0]
    if benches_v:
        b = benches_v[0]; perts = ["viewpoint", "lighting", "sensor_noise"]
        xs = np.arange(len(perts))
        for m, off in [("baseline", -w/2), ("ph", w/2)]:
            vals = [(sr[b].get(m, {}).get(p, {}).get("average") or 0)*100 for p in perts]
            ax.bar(xs+off, vals, w, label=m, color=C[m])
        ax.set_xticks(xs); ax.set_xticklabels(perts, fontsize=8); ax.set_ylim(0, 100)
        ax.set_title(f"{b} by perturbation"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    # (4) latency
    ax = axes[1, 1]
    lat = cmp.get("latency", {})
    if lat:
        ms = [lat["baseline"]["mean_ms"], lat["ph"]["mean_ms"]]
        ax.bar(["baseline", "ph"], ms, color=[C["baseline"], C["ph"]])
        for i, v in enumerate(ms):
            ax.text(i, v, f"{v:.1f}ms", ha="center", va="bottom")
        ax.set_title("Inference latency (parity)"); ax.set_ylabel("ms"); ax.grid(alpha=0.3, axis="y")
    else:
        ax.set_axis_off()
    fig.suptitle("SmolVLA + Persistent Homology — headline results", fontsize=14)
    _save(fig, outdir, "headline_summary")


def _save(fig, outdir, name):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(Path(outdir) / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[plots] {name}.png / .pdf")


def build_summary_pdf(sr, cmp, outdir, pdf_path):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        print(f"[plots] reportlab unavailable ({e}); skipping summary.pdf"); return
    c = canvas.Canvas(pdf_path, pagesize=A4); W, H = A4
    c.setFont("Helvetica-Bold", 15); c.drawString(2*cm, H-2*cm, "SmolVLA + Persistent Homology — Summary")
    c.setFont("Helvetica", 10); y = H-3*cm
    summ = sr.get("_summary", {})
    c.drawString(2*cm, y, "Per-model averages:"); y -= 0.6*cm
    for m in ("baseline", "ph"):
        for k, v in (summ.get(m, {}) or {}).items():
            c.drawString(2.4*cm, y, f"{m}  {k} = {v*100:.1f}%" if isinstance(v,(int,float)) else f"{m} {k}=n/a")
            y -= 0.5*cm
    if cmp.get("latency"):
        lb, lp = cmp["latency"]["baseline"]["mean_ms"], cmp["latency"]["ph"]["mean_ms"]
        y -= 0.3*cm; c.drawString(2*cm, y, f"Inference latency: baseline={lb:.1f}ms ph={lp:.1f}ms (PH train-only -> parity)")
    c.showPage()
    for png in sorted(Path(outdir).glob("*.png")):
        img = ImageReader(str(png)); iw, ih = img.getSize()
        scale = (W-4*cm)/iw; dh = ih*scale
        c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, H-2*cm, png.stem.replace("_", " "))
        c.drawImage(img, 2*cm, H-2.6*cm-dh, width=W-4*cm, height=dh); c.showPage()
    c.save(); print(f"[plots] wrote {pdf_path}")


def parse_runs(s):
    """ 'SUITE:variant=dir,...' -> {suite:{variant:dir}} """
    runs = {}
    if not s:
        return runs
    for part in s.split(","):
        key, d = part.split("=")
        suite, variant = key.split(":")
        runs.setdefault(suite, {})[variant] = d
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/success_rates.json")
    ap.add_argument("--compare", default="results/compare.json")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--summary", default="summary.pdf")
    ap.add_argument("--runs", default="LIBERO-10:baseline=outputs/baseline_full,LIBERO-10:ph=outputs/ph_full")
    args = ap.parse_args()
    sr = load_json(args.results); cmp = load_json(args.compare)
    runs = parse_runs(args.runs)
    training_curves(runs, "epoch", args.out)
    training_curves(runs, "step", args.out)
    indist_bars(sr, args.out)
    perturbation_bars(sr, args.out)
    params_latency(cmp, args.out)
    headline(sr, cmp, args.out)
    build_summary_pdf(sr, cmp, args.out, args.summary)
    print("[plots] DONE")


if __name__ == "__main__":
    main()
