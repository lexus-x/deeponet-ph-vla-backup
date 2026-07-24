#!/usr/bin/env python
"""Six plots for the v1 (mean-pool, NO cross-attention) comparison
(M1 flow | M3 DeepONet-v1 | M4 DeepONet+PH-v1): accuracy, robustness total,
latency, parameters, per-task success, robustness-by-perturbation.
Latency/params use the same MEASURED gate values as the v2 deck. Pure CPU
(reads existing eval JSONs only). Saves to DeepONet_Results/plots_v1/."""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("DeepONet_Results/plots_v1"); OUT.mkdir(parents=True, exist_ok=True)
LAB = ["M1 flow", "M3 DeepONet-v1", "M4 DeepONet+PH-v1"]
COL = ["#2E86DE", "#8E44AD", "#E67E22"]
CATS = ["Camera Viewpoints","Light Conditions","Sensor Noise","Background Textures",
        "Objects Layout","Robot Initial States","Language Instructions"]
CSHORT = ["Camera","Lighting","Sensor","Background","Layout","RobotInit","Language"]

ind = json.load(open("runs/eval_indist/success_rates.json"))["LIBERO-SPATIAL"]
pl  = json.load(open("runs/eval_plus/robustness_plus.json"))
PFX = ["m1", "m3", "m4"]   # flow / deeponet-v1 / deeponet+PH-v1

def in_seedavgs(pfx): return [ind[f"{pfx}_s{s}"]["average"]*100 for s in range(5) if f"{pfx}_s{s}" in ind]
def pl_seedavgs(pfx): return [pl[f"{pfx}_s{s}"]["robustness_average"]*100 for s in range(5)
                              if f"{pfx}_s{s}" in pl and pl[f"{pfx}_s{s}"].get("robustness_average") is not None]
def cat_seedavgs(pfx, cat): return [pl[f"{pfx}_s{s}"][cat]["average"]*100 for s in range(5)
                              if f"{pfx}_s{s}" in pl and cat in pl[f"{pfx}_s{s}"]]

acc = [in_seedavgs(p) for p in PFX]
rob = [pl_seedavgs(p) for p in PFX]
NSEED = [len(a) for a in acc]

def bar(vals, title, ylabel, fname, unit="%", ymax=100):
    m=[np.mean(v) for v in vals]; s=[np.std(v) for v in vals]
    fig,ax=plt.subplots(figsize=(7,5)); x=np.arange(len(vals))
    ax.bar(x,m,yerr=s,capsize=7,color=COL)
    for i,v in enumerate(m): ax.text(i,v+(ymax*0.015),f"{v:.1f}{unit}",ha="center",fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(LAB,fontsize=9); ax.set_ylim(0,ymax)
    ax.set_ylabel(ylabel); ax.set_title(title,fontweight="bold"); ax.grid(alpha=.25,axis="y")
    fig.tight_layout(); fig.savefig(OUT/fname,dpi=150); plt.close(fig)

# 1 accuracy, 2 robustness total
bar(acc, f"In-distribution accuracy (LIBERO-Spatial, seeds {NSEED})", "success rate (%)", "1_accuracy.png")
bar(rob, f"Robustness (LIBERO-Plus, 7 perturbations, seeds {NSEED})", "success rate (%)", "2_robustness.png")

# 3 latency (measured), 4 parameters (measured).  v1 and v1+PH share the 2.31M head -> same latency.
lat=[148.1, 23.3, 23.3]; par=[99.9, 2.31, 2.31]
for vals,fname,ylab,ttl,fmt,pad in [
    (lat,"3_latency.png","latency / action chunk (ms)","Inference latency (measured, batch 1) — lower is better","{:.0f} ms",3),
    (par,"4_parameters.png","action-head parameters (M)","Action-head size (active params) — lower is better","{:.1f}M",2)]:
    fig,ax=plt.subplots(figsize=(7,5)); x=np.arange(3); ax.bar(x,vals,color=COL)
    for i,v in enumerate(vals): ax.text(i,v+pad,fmt.format(v),ha="center",fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(LAB,fontsize=9); ax.set_ylabel(ylab)
    ax.set_title(ttl,fontweight="bold"); ax.grid(alpha=.25,axis="y")
    fig.tight_layout(); fig.savefig(OUT/fname,dpi=150); plt.close(fig)

# 5 success per task
def pertask(pfx):
    tab={}
    for s in range(5):
        for t,v in ind.get(f"{pfx}_s{s}",{}).get("per_task",{}).items():
            tab.setdefault(int(t),[]).append(v["success_rate"])
    return tab
tabs=[pertask(p) for p in PFX]; tids=sorted(tabs[0])
fig,ax=plt.subplots(figsize=(13,5.5)); x=np.arange(len(tids)); w=.26
for i in range(3):
    m=[100*np.mean(tabs[i][t]) for t in tids]; e=[100*np.std(tabs[i][t]) for t in tids]
    ax.bar(x+(i-1)*w,m,w,yerr=e,capsize=2,label=LAB[i],color=COL[i])
ax.set_xticks(x); ax.set_xticklabels([f"task{t}" for t in tids],fontsize=9)
ax.set_ylim(0,105); ax.set_ylabel("success rate (%)"); ax.legend()
ax.set_title("Success rate per task (mean ± std)",fontweight="bold"); ax.grid(alpha=.25,axis="y")
fig.tight_layout(); fig.savefig(OUT/"5_success_per_task.png",dpi=150); plt.close(fig)

# 6 robustness by perturbation (per LIBERO-Plus category)
fig,ax=plt.subplots(figsize=(13,5.5)); x=np.arange(len(CATS)); w=.26
for i,p in enumerate(PFX):
    m=[np.mean(cat_seedavgs(p,c)) for c in CATS]; e=[np.std(cat_seedavgs(p,c)) for c in CATS]
    ax.bar(x+(i-1)*w,m,w,yerr=e,capsize=2,label=LAB[i],color=COL[i])
ax.set_xticks(x); ax.set_xticklabels(CSHORT,fontsize=9)
ax.set_ylim(0,max(60,ax.get_ylim()[1])); ax.set_ylabel("success rate (%)"); ax.legend()
ax.set_title("Robustness by perturbation type (LIBERO-Plus)",fontweight="bold"); ax.grid(alpha=.25,axis="y")
fig.tight_layout(); fig.savefig(OUT/"6_robustness_by_perturbation.png",dpi=150); plt.close(fig)

print("saved 6 plots ->", OUT)
for pp in sorted(OUT.glob("*.png")): print("  ", pp.name)
print("seed counts (flow,v1,v1+PH):", NSEED)
