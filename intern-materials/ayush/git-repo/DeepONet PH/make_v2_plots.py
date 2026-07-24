#!/usr/bin/env python
"""Five separate plots for the v2 comparison (M1 flow | M3 DeepONet-v2 |
M4 DeepONet+PH-v2): accuracy, robustness, latency, parameters, per-task success.
Latency/params use MEASURED gate values. Saves to DeepONet_Results/plots_v2/."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("DeepONet_Results/plots_v2"); OUT.mkdir(parents=True, exist_ok=True)
LAB = ["M1 flow", "M3 DeepONet-v2", "M4 DeepONet+PH-v2"]
COL = ["#2E86DE", "#10AC84", "#F39C12"]
CATS = ["Camera Viewpoints","Light Conditions","Sensor Noise","Background Textures",
        "Objects Layout","Robot Initial States","Language Instructions"]

flow_in = json.load(open("runs/eval_indist/success_rates.json"))["LIBERO-SPATIAL"]
v2_in   = json.load(open("v2/runs/eval_v2_indist/success_rates.json"))["LIBERO-SPATIAL"]
flow_pl = json.load(open("runs/eval_plus/robustness_plus.json"))
v2_pl   = json.load(open("v2/runs/eval_v2_plus/robustness_plus.json"))

def in_seedavgs(src, pfx): return [src[f"{pfx}_s{s}"]["average"]*100 for s in range(5) if f"{pfx}_s{s}" in src]
def pl_seedavgs(src, pfx): return [src[f"{pfx}_s{s}"]["robustness_average"]*100 for s in range(5) if f"{pfx}_s{s}" in src and src[f"{pfx}_s{s}"].get("robustness_average") is not None]

acc = [in_seedavgs(flow_in,"m1"), in_seedavgs(v2_in,"m3v2"), in_seedavgs(v2_in,"m4v2")]
rob = [pl_seedavgs(flow_pl,"m1"), pl_seedavgs(v2_pl,"m3v2"), pl_seedavgs(v2_pl,"m4v2")]

def bar(vals, title, ylabel, fname, pct=True, ymax=100):
    m=[np.mean(v) for v in vals]; s=[np.std(v) for v in vals]
    fig,ax=plt.subplots(figsize=(7,5)); x=np.arange(3)
    ax.bar(x,m,yerr=s,capsize=7,color=COL)
    for i,v in enumerate(m): ax.text(i,v+(ymax*0.015),f"{v:.1f}"+("%" if pct else ""),ha="center",fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(LAB,fontsize=9); ax.set_ylim(0,ymax)
    ax.set_ylabel(ylabel); ax.set_title(title,fontweight="bold"); ax.grid(alpha=.25,axis="y")
    fig.tight_layout(); fig.savefig(OUT/fname,dpi=150); plt.close(fig)

# 1 accuracy, 2 robustness
bar(acc,"In-distribution accuracy (LIBERO-Spatial, 5 seeds)","success rate (%)","1_accuracy.png")
bar(rob,"Robustness (LIBERO-Plus, 7 perturbations, 5 seeds)","success rate (%)","2_robustness.png")

# 3 latency (measured), 4 parameters (measured)
# M3 DeepONet-v2 and M4 DeepONet+PH-v2 share the SAME v2 head (PH is a training-only
# loss, no extra inference params) -> identical 10.40M / 29.5ms. (flow = 99.9M / 148.1ms)
lat=[148.1,29.5,29.5]; par=[99.9,10.40,10.40]
fig,ax=plt.subplots(figsize=(7,5)); x=np.arange(3)
ax.bar(x,lat,color=COL)
for i,v in enumerate(lat): ax.text(i,v+3,f"{v:.0f} ms",ha="center",fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(LAB,fontsize=9); ax.set_ylabel("latency / action chunk (ms)")
ax.set_title("Inference latency (measured, batch 1)  — lower is better",fontweight="bold")
ax.grid(alpha=.25,axis="y"); fig.tight_layout(); fig.savefig(OUT/"3_latency.png",dpi=150); plt.close(fig)

fig,ax=plt.subplots(figsize=(7,5))
ax.bar(x,par,color=COL)
for i,v in enumerate(par): ax.text(i,v+2,f"{v:.1f}M",ha="center",fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(LAB,fontsize=9); ax.set_ylabel("action-head parameters (M)")
ax.set_title("Action-head size (active params) — lower is better",fontweight="bold")
ax.grid(alpha=.25,axis="y"); fig.tight_layout(); fig.savefig(OUT/"4_parameters.png",dpi=150); plt.close(fig)

# 5 success rate per task (seed mean)
def pertask(src,pfx):
    tab={}
    for s in range(5):
        for t,v in src.get(f"{pfx}_s{s}",{}).get("per_task",{}).items():
            tab.setdefault(int(t),[]).append(v["success_rate"])
    return tab
tabs=[pertask(flow_in,"m1"),pertask(v2_in,"m3v2"),pertask(v2_in,"m4v2")]
tids=sorted(tabs[0]); fig,ax=plt.subplots(figsize=(13,5.5)); x=np.arange(len(tids)); w=.26
for i in range(3):
    m=[100*np.mean(tabs[i][t]) for t in tids]; e=[100*np.std(tabs[i][t]) for t in tids]
    ax.bar(x+(i-1)*w,m,w,yerr=e,capsize=2,label=LAB[i],color=COL[i])
ax.set_xticks(x); ax.set_xticklabels([f"task{t}" for t in tids],fontsize=9)
ax.set_ylim(0,105); ax.set_ylabel("success rate (%)"); ax.legend()
ax.set_title("Success rate per task (mean ± std over 5 seeds)",fontweight="bold")
ax.grid(alpha=.25,axis="y"); fig.tight_layout(); fig.savefig(OUT/"5_success_per_task.png",dpi=150); plt.close(fig)

print("saved 5 plots ->", OUT)
for p in sorted(OUT.glob("*.png")): print("  ", p.name)
