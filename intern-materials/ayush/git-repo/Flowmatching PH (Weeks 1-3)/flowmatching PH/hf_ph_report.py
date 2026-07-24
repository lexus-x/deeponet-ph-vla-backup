#!/usr/bin/env python
"""
hf_ph_report.py
===============
Aggregate the PH-finetune-on-official-checkpoint results (object accuracy +
one perturbation robustness, across lambda + a flow-only control) and produce:
  * cross-lambda: accuracy vs lambda + robustness vs lambda (control = dashed ref)
  * per-lambda: per-task object success (control vs lambda) and per-task robustness
  * latency + parameters (constant across lambda; from compare.json)
  * summary.json + summary.pdf
All into <root>/plots and <root>/summary.*.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CB, CP = "#1f77b4", "#d62728"


def tag_to_lambda(tag):
    if tag == "control":
        return 0.0
    return float(tag.replace("lambda_", "").replace("p", "."))


def load_model(root, tag, pert):
    f = Path(root) / tag / "results" / "success_rates.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    obj = d.get("LIBERO-OBJECT", {}).get("ph", {})
    v = d.get("LIBERO-OBJECT-V", {}).get("ph", {}).get(pert, {})
    return {
        "obj_avg": (obj.get("average") or 0) * 100,
        "obj_tasks": {int(k): val["success_rate"] * 100 for k, val in obj.get("per_task", {}).items()},
        "rob_avg": (v.get("average") or 0) * 100,
        "rob_tasks": {int(k): val["success_rate"] * 100 for k, val in v.get("per_task", {}).items()},
    }


def save(fig, outdir, name):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(Path(outdir) / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[hfph-report] {name}.png/.pdf")


def per_task_fig(ctrl, mod, lam, kind, outdir):
    ck = ctrl[f"{kind}_tasks"]; mk = mod[f"{kind}_tasks"]
    tasks = sorted(set(ck) | set(mk))
    x = np.arange(len(tasks)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.bar(x - w/2, [ck.get(t, np.nan) for t in tasks], w,
           label=f"control (avg {ctrl[kind+'_avg']:.1f}%)", color=CB)
    ax.bar(x + w/2, [mk.get(t, np.nan) for t in tasks], w,
           label=f"PH λ={lam} (avg {mod[kind+'_avg']:.1f}%)", color=CP)
    ax.set_xticks(x); ax.set_xticklabels([f"t{t}" for t in tasks]); ax.set_ylim(0, 105)
    ax.set_ylabel("success rate (%)")
    title = "object accuracy" if kind == "obj" else "viewpoint robustness"
    ax.set_title(f"LIBERO-Object per-task {title}: control vs PH λ={lam}")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    save(fig, outdir, f"per_task_{kind}_lambda_{str(lam).replace('.','p')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--pert", default="viewpoint")
    args = ap.parse_args()
    root = Path(args.root); plots = root / "plots"

    tags = ["control", "lambda_0p02", "lambda_0p05", "lambda_0p1", "lambda_0p2", "lambda_0p5"]
    data = {t: load_model(root, t, args.pert) for t in tags}
    data = {t: v for t, v in data.items() if v is not None}
    if "control" not in data:
        print("[hfph-report] no control results; aborting"); return
    ctrl = data["control"]

    # ---- cross-lambda accuracy + robustness ----
    lam_tags = [t for t in tags if t != "control" and t in data]
    lams = [tag_to_lambda(t) for t in lam_tags]
    acc = [data[t]["obj_avg"] for t in lam_tags]
    rob = [data[t]["rob_avg"] for t in lam_tags]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(lams, acc, "o-", color=CB, label="PH object accuracy")
    ax.plot(lams, rob, "s-", color=CP, label=f"PH robustness ({args.pert})")
    ax.axhline(ctrl["obj_avg"], ls="--", color=CB, alpha=0.6, label=f"control accuracy ({ctrl['obj_avg']:.1f}%)")
    ax.axhline(ctrl["rob_avg"], ls="--", color=CP, alpha=0.6, label=f"control robustness ({ctrl['rob_avg']:.1f}%)")
    ax.set_xscale("log"); ax.set_xlabel("lambda_ph (log)"); ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100); ax.set_title("PH on official smolvla_libero (LIBERO-Object)\npoints above dashed = PH beats control")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for l, a, r in zip(lams, acc, rob):
        ax.annotate(f"{a:.0f}", (l, a), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
        ax.annotate(f"{r:.0f}", (l, r), textcoords="offset points", xytext=(0, -12), fontsize=7, ha="center")
    save(fig, plots, "cross_lambda_accuracy_robustness")

    # ---- per-lambda per-task (object + robustness) ----
    for t in lam_tags:
        lam = tag_to_lambda(t)
        per_task_fig(ctrl, data[t], lam, "obj", plots)
        per_task_fig(ctrl, data[t], lam, "rob", plots)

    # ---- latency + params (constant) ----
    cmp = json.loads(Path(args.compare).read_text()) if args.compare and Path(args.compare).exists() else {}
    if cmp:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
        p = cmp.get("params", {}).get("baseline", {})
        labs = [k for k in p if k != "TOTAL"]
        a1.barh(labs, [p[k]/1e6 for k in labs], color="#4c72b0")
        a1.set_xlabel("params (M)"); a1.set_title(f"Parameters (identical all λ; TOTAL={p.get('TOTAL',0)/1e6:.1f}M)")
        a1.grid(alpha=0.3, axis="x")
        lat = cmp.get("latency", {})
        if lat:
            ms = [lat["baseline"]["mean_ms"], lat["ph"]["mean_ms"]]
            a2.bar(["control", "PH"], ms, color=[CB, CP])
            for i, vv in enumerate(ms): a2.text(i, vv, f"{vv:.1f}ms", ha="center", va="bottom")
            a2.set_ylabel("latency (ms, batch=1)"); a2.set_title("Inference latency (PH train-only → identical all λ)")
            a2.grid(alpha=0.3, axis="y")
        save(fig, plots, "latency_params")

    # ---- summary ----
    summary = {"control": {"object_acc": ctrl["obj_avg"], f"robust_{args.pert}": ctrl["rob_avg"]},
               "ph_by_lambda": {str(tag_to_lambda(t)): {"object_acc": data[t]["obj_avg"],
                                f"robust_{args.pert}": data[t]["rob_avg"]} for t in lam_tags}}
    if cmp.get("latency"):
        summary["latency_ms"] = {"control": cmp["latency"]["baseline"]["mean_ms"], "ph": cmp["latency"]["ph"]["mean_ms"]}
    if cmp.get("params"):
        summary["total_params_M"] = cmp["params"]["baseline"]["TOTAL"]/1e6
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[hfph-report] control: acc=%.1f rob=%.1f" % (ctrl["obj_avg"], ctrl["rob_avg"]))
    for t in lam_tags:
        print(f"[hfph-report] λ={tag_to_lambda(t):<5} acc={data[t]['obj_avg']:.1f}  rob={data[t]['rob_avg']:.1f}")

    # summary pdf
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        c = canvas.Canvas(str(root / "summary.pdf"), pagesize=A4); W, H = A4
        c.setFont("Helvetica-Bold", 14); c.drawString(2*cm, H-2*cm, "PH on official SmolVLA-LIBERO (Object)")
        c.setFont("Helvetica", 10); y = H-3*cm
        c.drawString(2*cm, y, f"control (no PH): acc={ctrl['obj_avg']:.1f}%  robust({args.pert})={ctrl['rob_avg']:.1f}%"); y -= 0.6*cm
        for t in lam_tags:
            c.drawString(2*cm, y, f"PH λ={tag_to_lambda(t)}: acc={data[t]['obj_avg']:.1f}%  robust={data[t]['rob_avg']:.1f}%"); y -= 0.5*cm
        c.showPage()
        for png in sorted(plots.glob("*.png")):
            img = ImageReader(str(png)); iw, ih = img.getSize(); sc = (W-4*cm)/iw
            c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, H-2*cm, png.stem)
            c.drawImage(img, 2*cm, H-2.5*cm-ih*sc, width=W-4*cm, height=ih*sc); c.showPage()
        c.save(); print(f"[hfph-report] wrote {root}/summary.pdf")
    except Exception as e:
        print(f"[hfph-report] summary.pdf skipped ({e})")


if __name__ == "__main__":
    main()
