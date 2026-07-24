#!/usr/bin/env python
"""
make_week3_folder.py
====================
Builds an organized, zippable asset folder 'week 3 ppt/' with subfolders
v1 / stronger run / sweep / hugging face, containing bar charts (accuracy +
robustness on all 3 perturbations), the PH-success & flow-matching-failure
videos, the lambda tables, and lambda-vs-success / lambda-vs-robustness charts
(LINEAR lambda axis). Then zips it. Re-runnable.
"""
import json, os, shutil, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/user/Desktop/Ayush PH test")
OUT = ROOT / "week 3 ppt"
BLUE, RED, GREEN, PURPLE, ORANGE, YELLOW = "#2E86DE", "#EE5253", "#10AC84", "#5F27CD", "#FF9F43", "#FECA57"
PERTS = ["viewpoint", "lighting", "sensor_noise"]


def load(p):
    p = Path(p); return json.load(open(p)) if p.exists() else None


def pct(x): return round(x*100, 1) if isinstance(x, (int, float)) else None


# ---------------------------------------------------------------- bar charts
def grouped_bar(path, title, groups, series_labels=("flow-matching", "flow + PH"),
                colors=(BLUE, RED), ylabel="success rate (%)"):
    """groups: list of (label, val_baseline, val_ph)"""
    fig, ax = plt.subplots(figsize=(8.5, 4.8)); x = np.arange(len(groups)); w = 0.36
    ax.bar(x-w/2, [g[1] for g in groups], w, label=series_labels[0], color=colors[0])
    ax.bar(x+w/2, [g[2] for g in groups], w, label=series_labels[1], color=colors[1])
    for i, g in enumerate(groups):
        for off, v in [(-w/2, g[1]), (w/2, g[2])]:
            if v is not None:
                ax.text(i+off, v+1, f"{v:.0f}", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=11)
    ax.set_ylim(0, 100); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=13, weight="bold", color=PURPLE)
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- table image
def table_image(path, title, headers, rows, best_idx):
    fig, ax = plt.subplots(figsize=(9.5, 0.7+0.5*len(rows))); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 1.6)
    for j in range(len(headers)):
        c = tbl[0, j]; c.set_facecolor(PURPLE); c.set_text_props(color="white", weight="bold")
    for i in range(len(rows)):
        for j in range(len(headers)):
            c = tbl[i+1, j]
            if i == best_idx: c.set_facecolor(YELLOW); c.set_text_props(weight="bold")
            elif i == 0: c.set_facecolor("#EEF2FF")
    ax.set_title(title, fontsize=14, weight="bold", color=PURPLE, pad=14)
    fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- lambda line (LINEAR)
def lambda_line(path, title, lams, vals, base, ylabel, color, best_lam=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(lams, vals, "o-", color=color, lw=2.5, ms=10)
    ax.axhline(base, ls="--", color="#888", lw=2, label=f"baseline (λ=0): {base:.0f}%")
    for l, v in zip(lams, vals):
        if v is not None:
            ax.annotate(f"{v:.0f}", (l, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10, weight="bold")
    if best_lam in lams:
        bi = lams.index(best_lam); ax.scatter([best_lam], [vals[bi]], s=320, facecolors="none", edgecolors=GREEN, lw=3, zorder=5)
    ax.set_xlabel("λ (lambda)", fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 100); ax.set_title(title, fontsize=13, weight="bold", color=PURPLE)
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(path, bbox_inches="tight", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- data loaders
def sweep_rows_p1():
    strong = load("output/results_strong/success_rates.json")
    def get(ind, v): return (pct(ind["average"]), pct(v["viewpoint"]["average"]),
                             pct(v["lighting"]["average"]), pct(v["sensor_noise"]["average"]),
                             pct(v["average_over_perturbations"]))
    d = {0.0: get(strong["LIBERO-SPATIAL"]["baseline"], strong["LIBERO-SPATIAL-V"]["baseline"]),
         0.1: get(strong["LIBERO-SPATIAL"]["ph"], strong["LIBERO-SPATIAL-V"]["ph"])}
    for lam, t in [(0.02, "0p02"), (0.05, "0p05"), (0.2, "0p2"), (0.5, "0p5")]:
        j = load(f"output/results_sweep/lambda_{t}.json")
        if j: d[lam] = get(j["LIBERO-SPATIAL"]["ph"], j["LIBERO-SPATIAL-V"]["ph"])
    return dict(sorted(d.items()))


def sweep_rows_p2():
    HF = "hugging face ckp/ph_object"; d = {}
    for lam, t in [(0.0, "control"), (0.02, "lambda_0p02"), (0.05, "lambda_0p05"),
                   (0.1, "lambda_0p1"), (0.2, "lambda_0p2"), (0.5, "lambda_0p5")]:
        j = load(f"{HF}/{t}/results/success_rates.json")
        if not j: continue
        v = j["LIBERO-OBJECT-V"]["ph"]
        d[lam] = (pct(j["LIBERO-OBJECT"]["ph"]["average"]), pct(v.get("viewpoint", {}).get("average")),
                  pct(v.get("lighting", {}).get("average")), pct(v.get("sensor_noise", {}).get("average")),
                  pct(v.get("average_over_perturbations")))
    return d


def best_lambda(d):
    best, bv = None, -1
    for lam, r in d.items():
        if lam == 0: continue
        m = r[4] if r[4] is not None else r[1]
        if m is not None and m > bv: bv, best = m, lam
    return best


def fmt(x): return f"{x:.1f}" if isinstance(x, (int, float)) else "—"


def make_table_and_charts(folder, data, title, suite_lbl):
    lams = list(data.keys())
    rows = [["baseline" if l == 0 else f"{l}", fmt(data[l][0]), fmt(data[l][1]),
             fmt(data[l][2]), fmt(data[l][3]), fmt(data[l][4])] for l in lams]
    bl = best_lambda(data); bi = lams.index(bl)
    table_image(folder/"lambda_table_accuracy_robustness.png", f"{suite_lbl} — λ table (best λ={bl})",
                ["λ", "Accuracy", "Viewpoint-V", "Lighting-V", "Sensor-V", "Robust avg"], rows, bi)
    ll = [l for l in lams if l > 0]
    acc = [data[l][0] for l in ll]
    rob = [data[l][4] if data[l][4] is not None else data[l][1] for l in ll]
    lambda_line(folder/"lambda_vs_success_rate.png", f"{suite_lbl}: λ vs Success rate",
                ll, acc, data[0.0][0], "success rate (%)", BLUE, bl)
    lambda_line(folder/"lambda_vs_robustness.png", f"{suite_lbl}: λ vs Robustness",
                ll, rob, data[0.0][4] if data[0.0][4] is not None else data[0.0][1], "robustness (%)", RED, bl)


# ====================================================== build
shutil.rmtree(OUT, ignore_errors=True)
for sub in ["v1", "stronger run", "sweep", "hugging face"]:
    (OUT/sub).mkdir(parents=True, exist_ok=True)

# ---- V1 (LIBERO-10) ----
v1 = load("output/v1/results/success_rates.json")
grouped_bar(OUT/"v1/accuracy_baseline_vs_ph.png", "V1 — LIBERO-10 accuracy (success rate)",
            [("LIBERO-10", pct(v1["LIBERO-10"]["baseline"]["average"]), pct(v1["LIBERO-10"]["ph"]["average"]))])
vb, vp = v1["LIBERO-V"]["baseline"], v1["LIBERO-V"]["ph"]
grouped_bar(OUT/"v1/robustness_3perturbations.png", "V1 — robustness on 3 perturbations",
            [(p.replace("_", " "), pct(vb[p]["average"]), pct(vp[p]["average"])) for p in PERTS])

# ---- STRONGER RUN (LIBERO-10 + Spatial) ----
st = load("output/results_strong/success_rates.json")
grouped_bar(OUT/"stronger run/accuracy_baseline_vs_ph.png", "Stronger run — accuracy (success rate), λ=0.1",
            [("LIBERO-10", pct(st["LIBERO-10"]["baseline"]["average"]), pct(st["LIBERO-10"]["ph"]["average"])),
             ("Spatial", pct(st["LIBERO-SPATIAL"]["baseline"]["average"]), pct(st["LIBERO-SPATIAL"]["ph"]["average"]))])
for suite, key in [("LIBERO-10", "LIBERO-10-V"), ("Spatial", "LIBERO-SPATIAL-V")]:
    b, p = st[key]["baseline"], st[key]["ph"]
    grouped_bar(OUT/f"stronger run/robustness_{suite}_3perturbations.png",
                f"Stronger run — {suite} robustness on 3 perturbations (λ=0.1)",
                [(pp.replace("_", " "), pct(b[pp]["average"]), pct(p[pp]["average"])) for pp in PERTS])
# videos (PH success + flow-matching failure)
vsrc = "output/videos_strong/spatial/indist"
for src, dst in [(f"{vsrc}/task04_ph.mp4", "stronger run/PH_success_spatial_task4.mp4"),
                 (f"{vsrc}/task04_baseline.mp4", "stronger run/flowmatching_failure_spatial_task4.mp4")]:
    if Path(src).exists(): shutil.copy(src, OUT/dst)

# ---- SWEEP (P1 - our trained model, Spatial) ----
make_table_and_charts(OUT/"sweep", sweep_rows_p1(), "Sweep", "LIBERO-Spatial (our trained model)")

# ---- HUGGING FACE (P2 - official checkpoint, Object) ----
make_table_and_charts(OUT/"hugging face", sweep_rows_p2(), "HF", "LIBERO-Object (official checkpoint)")

# ---- zip ----
zip_path = ROOT/"week 3 ppt.zip"
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in OUT.rglob("*"):
        if f.is_file(): z.write(f, f.relative_to(ROOT))
print("folder:", OUT)
for f in sorted(OUT.rglob("*")):
    if f.is_file(): print("  ", f.relative_to(OUT))
print("\nzip:", zip_path, f"({zip_path.stat().st_size/1e6:.1f} MB)")
