#!/usr/bin/env python
"""
aggregate_final.py
==================
Final cross-model aggregation for the whole campaign:
  flow (M1) | DeepONet-v1 | DeepONet-v1+PH | DeepONet-v2 | DeepONet-v2+PH
pulling in-dist + LIBERO-Plus robustness from BOTH the parent eval dirs (v1/flow)
and the v2/ eval dirs (v2). Produces mean+-std over seeds, paired significance
tests, error-bar plots, a machine-readable report.json, and a human-readable
REPORT.md. Robust to partial data (skips missing models/files) so it can run any
time, including mid-campaign.

Run from the v2/ dir (default paths assume that), or pass explicit paths.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from scipy import stats as _st
except Exception:
    _st = None

CATS = ["Camera Viewpoints", "Light Conditions", "Sensor Noise", "Background Textures",
        "Objects Layout", "Robot Initial States", "Language Instructions"]

# (key, label, prefix, indist_json, plus_json, color)
def model_specs(parent, v2):
    pj_in = f"{parent}/runs/eval_indist/success_rates.json"
    pj_pl = f"{parent}/runs/eval_plus/robustness_plus.json"
    vj_in = f"{v2}/runs/eval_v2_indist/success_rates.json"
    vj_pl = f"{v2}/runs/eval_v2_plus/robustness_plus.json"
    return [
        ("flow",     "M1 flow",        "m1",   pj_in, pj_pl, "#2E86DE"),
        ("don_v1",   "M3 DeepONet v1", "m3",   pj_in, pj_pl, "#8E44AD"),
        ("donph_v1", "M4 DON+PH v1",   "m4",   pj_in, pj_pl, "#C39BD3"),
        ("don_v2",   "M3 DeepONet v2", "m3v2", vj_in, vj_pl, "#10AC84"),
        ("donph_v2", "M4 DON+PH v2",   "m4v2", vj_in, vj_pl, "#F39C12"),
    ]

def _load_json(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else {}

def load_indist(path, prefix):
    """-> {seed:{'avg':x,'per_task':{t:r}}} for models named <prefix>_s<seed>."""
    d = _load_json(path).get("LIBERO-SPATIAL", {})
    out = {}
    for name, rec in d.items():
        m = re.fullmatch(rf"{re.escape(prefix)}_s(\d+)", name)
        if m and isinstance(rec, dict) and "per_task" in rec:
            out[int(m.group(1))] = {"avg": rec.get("average"),
                                    "per_task": {k: v["success_rate"] for k, v in rec["per_task"].items()}}
    return out

def load_plus(path, prefix):
    d = _load_json(path); out = {}
    for name, rec in d.items():
        if name.startswith("_"): continue
        m = re.fullmatch(rf"{re.escape(prefix)}_s(\d+)", name)
        if m and isinstance(rec, dict):
            out[int(m.group(1))] = {"avg": rec.get("robustness_average"),
                                    "per_cat": {c: rec[c]["average"] for c in CATS
                                                if isinstance(rec.get(c), dict) and rec[c].get("average") is not None}}
    return out

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, None, 0)

def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b)); a, b = a[mask], b[mask]
    if len(a) < 2: return {"n": int(len(a)), "mean_diff": None, "t_p": None, "wilcoxon_p": None}
    r = {"n": int(len(a)), "mean_diff": float(np.mean(a - b))}
    if _st is not None:
        try: r["t_p"] = float(_st.ttest_rel(a, b).pvalue)
        except Exception: r["t_p"] = None
        try: r["wilcoxon_p"] = float(_st.wilcoxon(a, b).pvalue) if np.any(a - b) else 1.0
        except Exception: r["wilcoxon_p"] = None
    else: r["t_p"] = r["wilcoxon_p"] = None
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="..")
    ap.add_argument("--v2", default=".")
    ap.add_argument("--compare", default="../runs/compare/compare.json")
    ap.add_argument("--out", default="../runs/report_final")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    specs = model_specs(args.parent, args.v2)
    data = {}
    for key, label, prefix, ji, jp, color in specs:
        ind = load_indist(ji, prefix); pl = load_plus(jp, prefix)
        data[key] = {"label": label, "color": color, "indist": ind, "plus": pl}

    report = {"models": {}, "significance": {}}
    for key, d in data.items():
        am, asd, an = mean_std([d["indist"][s]["avg"] for s in d["indist"]])
        rm, rsd, rn = mean_std([d["plus"][s]["avg"] for s in d["plus"]])
        report["models"][key] = {"label": d["label"], "n_seeds_indist": an, "n_seeds_robust": rn,
                                 "indist_mean": am, "indist_std": asd,
                                 "robust_mean": rm, "robust_std": rsd}

    # significance: in-dist per-(task,seed); robust per-(cat,seed)
    def pairs_indist(ka, kb):
        a, b = [], []
        seeds = set(data[ka]["indist"]) & set(data[kb]["indist"])
        for s in seeds:
            pa, pb = data[ka]["indist"][s]["per_task"], data[kb]["indist"][s]["per_task"]
            for t in set(pa) & set(pb): a.append(pa[t]); b.append(pb[t])
        return a, b
    def pairs_plus(ka, kb):
        a, b = [], []
        seeds = set(data[ka]["plus"]) & set(data[kb]["plus"])
        for s in seeds:
            ca, cb = data[ka]["plus"][s]["per_cat"], data[kb]["plus"][s]["per_cat"]
            for c in set(ca) & set(cb): a.append(ca[c]); b.append(cb[c])
        return a, b
    for ka, kb in [("don_v2","flow"),("donph_v2","flow"),("don_v2","don_v1"),
                   ("donph_v2","don_v2"),("don_v1","flow")]:
        report["significance"][f"acc_{ka}_vs_{kb}"] = paired(*pairs_indist(ka, kb))
        report["significance"][f"rob_{ka}_vs_{kb}"] = paired(*pairs_plus(ka, kb))

    if Path(args.compare).exists():
        report["efficiency_gate"] = _load_json(args.compare)
    (out / "report.json").write_text(json.dumps(report, indent=2))

    keys = list(data.keys()); labels = [data[k]["label"] for k in keys]; colors = [data[k]["color"] for k in keys]
    # ---- accuracy + robustness bars
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, field, ttl in [(axes[0], "indist", "In-distribution accuracy"),
                           (axes[1], "robust", "Robustness (LIBERO-Plus)")]:
        xs = np.arange(len(keys))
        ms = [report["models"][k][f"{field}_mean"] or 0 for k in keys]
        sd = [report["models"][k][f"{field}_std"] or 0 for k in keys]
        ax.bar(xs, [100*m for m in ms], yerr=[100*s for s in sd], capsize=6, color=colors)
        for i, m in enumerate(ms):
            if m: ax.text(i, 100*m+1.5, f"{100*m:.1f}", ha="center", fontweight="bold", fontsize=9)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20, fontsize=8, ha="right")
        ax.set_ylim(0, 100); ax.set_ylabel("success (%)"); ax.set_title(ttl + "  (mean +/- std)", fontweight="bold")
        ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(out / "accuracy_robustness.png", dpi=150); plt.close(fig)

    # ---- per-task (flow, v1, v2 DeepONet) seed-mean
    trio = [("flow","M1 flow"),("don_v1","M3 v1"),("don_v2","M3 v2")]
    tids = sorted({t for k,_ in trio for s in data[k]["indist"] for t in data[k]["indist"][s]["per_task"]}, key=int)
    if tids:
        fig, ax = plt.subplots(figsize=(13,5.5)); x=np.arange(len(tids)); w=.26
        for i,(k,lb) in enumerate(trio):
            ms=[100*np.mean([data[k]["indist"][s]["per_task"][t] for s in data[k]["indist"] if t in data[k]["indist"][s]["per_task"]] or [0]) for t in tids]
            ax.bar(x+(i-1)*w, ms, w, label=lb, color=data[k]["color"])
        ax.set_xticks(x); ax.set_xticklabels([f"t{t}" for t in tids]); ax.set_ylim(0,100)
        ax.set_ylabel("success (%)"); ax.legend(); ax.grid(alpha=.25,axis="y")
        ax.set_title("Per-task in-dist: flow vs DeepONet-v1 vs DeepONet-v2 (seed mean)", fontweight="bold")
        fig.tight_layout(); fig.savefig(out/"per_task_v1_v2.png", dpi=150); plt.close(fig)

    # ---- robustness per category (all models)
    fig, ax = plt.subplots(figsize=(14,6)); x=np.arange(len(CATS)); w=.16
    for i,k in enumerate(keys):
        ms=[]
        for c in CATS:
            vals=[data[k]["plus"][s]["per_cat"].get(c) for s in data[k]["plus"] if c in data[k]["plus"][s]["per_cat"]]
            ms.append(100*np.mean(vals) if vals else 0)
        ax.bar(x+(i-2)*w, ms, w, label=data[k]["label"], color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels([c.replace(" ","\n") for c in CATS], fontsize=8); ax.set_ylim(0,100)
    ax.set_ylabel("success (%)"); ax.legend(fontsize=8); ax.grid(alpha=.25,axis="y")
    ax.set_title("LIBERO-Plus robustness per perturbation (mean over seeds)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out/"robustness_per_category.png", dpi=150); plt.close(fig)

    # ---- REPORT.md
    def f(m,s): return f"{100*m:.1f} ± {100*s:.1f}" if m is not None else "—"
    lines = ["# DeepONet vs Flow-matching — Final Report\n",
             "## Headline (LIBERO-Spatial, replan-5, mean ± std over seeds)\n",
             "| Model | In-dist acc (%) | Robustness (%) | Head params | Latency |",
             "|---|---|---|---|---|"]
    eff = report.get("efficiency_gate", {})
    def head_lat(key):
        if key == "flow" and eff: return f"{eff['flow']['params']['head']/1e6:.1f}M", f"{eff['flow']['latency_ms_mean']:.0f}ms"
        if key in ("don_v1","donph_v1") and eff: return f"{eff['deeponet']['params']['head']/1e6:.1f}M", f"{eff['deeponet']['latency_ms_mean']:.0f}ms"
        if key in ("don_v2","donph_v2"): return "~10.4M", "~25ms"
        return "—","—"
    for k in keys:
        mm=report["models"][k]; hp,lt=head_lat(k)
        lines.append(f"| {mm['label']} | {f(mm['indist_mean'],mm['indist_std'])} (n={mm['n_seeds_indist']}) | {f(mm['robust_mean'],mm['robust_std'])} (n={mm['n_seeds_robust']}) | {hp} | {lt} |")
    lines += ["\n## Paired significance (positive mean_diff favours first model)\n",
              "| comparison | metric | n | Δ (pp) | t p-value | Wilcoxon p |", "|---|---|---|---|---|---|"]
    for k,v in report["significance"].items():
        if v["mean_diff"] is None: continue
        metric = "in-dist" if k.startswith("acc") else "robust"
        comp = k.split("_",1)[1]
        tp = f"{v['t_p']:.3f}" if v['t_p'] is not None else "—"
        wp = f"{v['wilcoxon_p']:.3f}" if v['wilcoxon_p'] is not None else "—"
        lines.append(f"| {comp} | {metric} | {v['n']} | {100*v['mean_diff']:+.1f} | {tp} | {wp} |")
    lines += ["\n## Plots", "- accuracy_robustness.png", "- per_task_v1_v2.png", "- robustness_per_category.png",
              "\n_Significance via paired t-test + Wilcoxon over matched (task,seed) / (category,seed) units._"]
    (out / "REPORT.md").write_text("\n".join(lines))

    print("=== FINAL REPORT ===")
    for k in keys:
        mm=report["models"][k]
        print(f"{mm['label']:16s} in-dist {f(mm['indist_mean'],mm['indist_std']):>14s}  robust {f(mm['robust_mean'],mm['robust_std']):>14s}")
    print(f"saved -> {out}/ (report.json, REPORT.md, 3 plots)")

if __name__ == "__main__":
    main()
