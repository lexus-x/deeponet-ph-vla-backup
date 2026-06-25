#!/usr/bin/env python
"""Per-suite plots for the Object/Goal generalization runs (flow | M3 DeepONet-v2 |
M4 DeepONet+PH-v2), 1 seed. Reads a suite folder's eval JSONs, writes plots + CSVs.
Usage: python make_suite_plots.py --suite_dir ../Object --suite libero_object --label OBJECT
Pure CPU."""
import json, csv, argparse, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CATS=["Camera Viewpoints","Light Conditions","Sensor Noise","Background Textures",
      "Objects Layout","Robot Initial States","Language Instructions"]
CSHORT=["Camera","Lighting","Sensor","Background","Layout","RobotInit","Language"]
LAB=["M1 flow","M3 DeepONet-v2","M4 DeepONet+PH-v2"]; KEYS=["flow","m3","m4"]
COL=["#2E86DE","#10AC84","#F39C12"]

ap=argparse.ArgumentParser()
ap.add_argument("--suite_dir", required=True)
ap.add_argument("--suite", required=True)         # libero_object / libero_goal
ap.add_argument("--label", required=True)         # OBJECT / GOAL
ap.add_argument("--eval_subdir", default="eval_indist")  # eval_indist (3-model) or eval_flow (flow-only)
a=ap.parse_args()
SD=Path(a.suite_dir); OUT=SD/"plots"; DAT=SD/"data"; OUT.mkdir(parents=True,exist_ok=True); DAT.mkdir(parents=True,exist_ok=True)
BN=a.suite.replace("libero_","LIBERO-").upper()

def jload(p): p=Path(p); return json.loads(p.read_text()) if p.exists() else {}
ind=jload(SD/f"runs/{a.eval_subdir}/success_rates.json").get(BN,{})
pl =jload(SD/"runs/eval_plus/robustness_plus.json")

def acc(k):
    v=ind.get(k,{}).get("average"); return v*100 if v is not None else None
def rob(k):
    v=pl.get(k,{}).get("robustness_average"); return v*100 if v is not None else None
def cat(k,c):
    d=pl.get(k,{}).get(c); return d["average"]*100 if isinstance(d,dict) and d.get("average") is not None else None

accs=[acc(k) for k in KEYS]; robs=[rob(k) for k in KEYS]

def barplot(vals, title, fname, unit="%"):
    xs=[i for i,v in enumerate(vals) if v is not None]
    if not xs: print("  skip (no data):",fname); return
    fig,axx=plt.subplots(figsize=(7,5))
    axx.bar(range(3),[v or 0 for v in vals],color=COL)
    for i,v in enumerate(vals):
        if v is not None: axx.text(i,v+1.5,f"{v:.1f}{unit}",ha="center",fontweight="bold")
    axx.set_xticks(range(3)); axx.set_xticklabels(LAB,fontsize=9); axx.set_ylim(0,105)
    axx.set_ylabel("success rate (%)"); axx.set_title(title,fontweight="bold"); axx.grid(alpha=.25,axis="y")
    fig.tight_layout(); fig.savefig(OUT/fname,dpi=150); plt.close(fig); print("  saved",fname)

# 1 accuracy, 2 robustness
barplot(accs, f"In-distribution accuracy ({BN}, 1 seed)", "1_accuracy.png")
if any(r is not None for r in robs):
    barplot(robs, f"Robustness ({BN} LIBERO-Plus, 7 perturbations, 1 seed)", "2_robustness.png")

# 3 per-task accuracy (grouped flow/m3/m4)
def pertask(k):
    d=ind.get(k,{}).get("per_task",{}); return {int(t):v["success_rate"]*100 for t,v in d.items()}
pts=[pertask(k) for k in KEYS]
if pts[0]:
    tids=sorted(pts[0]); x=np.arange(len(tids)); w=.26
    fig,axx=plt.subplots(figsize=(13,5.5))
    for i in range(3):
        axx.bar(x+(i-1)*w,[pts[i].get(t,0) for t in tids],w,label=LAB[i],color=COL[i])
    axx.set_xticks(x); axx.set_xticklabels([f"task{t}" for t in tids],fontsize=9); axx.set_ylim(0,108)
    axx.set_ylabel("success rate (%)"); axx.legend(); axx.set_title(f"Success rate per task ({BN}, 1 seed)",fontweight="bold")
    axx.grid(alpha=.25,axis="y"); fig.tight_layout(); fig.savefig(OUT/"3_success_per_task.png",dpi=150); plt.close(fig); print("  saved 3_success_per_task.png")

# 4 robustness by perturbation
if any(cat(k,c) is not None for k in KEYS for c in CATS):
    x=np.arange(len(CATS)); w=.26; fig,axx=plt.subplots(figsize=(13,5.5))
    for i,k in enumerate(KEYS):
        axx.bar(x+(i-1)*w,[cat(k,c) or 0 for c in CATS],w,label=LAB[i],color=COL[i])
    axx.set_xticks(x); axx.set_xticklabels(CSHORT,fontsize=9); axx.set_ylim(0,max(60,axx.get_ylim()[1]))
    axx.set_ylabel("success rate (%)"); axx.legend(); axx.set_title(f"Robustness by perturbation ({BN})",fontweight="bold")
    axx.grid(alpha=.25,axis="y"); fig.tight_layout(); fig.savefig(OUT/"4_robustness_by_perturbation.png",dpi=150); plt.close(fig); print("  saved 4_robustness_by_perturbation.png")

# CSVs
with open(DAT/"summary.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["model","indist_acc","robustness"])
    for lab,k in zip(LAB,KEYS): w.writerow([lab, f"{acc(k):.1f}" if acc(k) is not None else "NA", f"{rob(k):.1f}" if rob(k) is not None else "NA"])
if pts[0]:
    with open(DAT/"success_per_task.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["task"]+LAB)
        for t in sorted(pts[0]): w.writerow([f"task{t}"]+[f"{pts[i].get(t,0):.1f}" for i in range(3)])
print(f"[{a.label}] plots+CSVs ->", OUT, DAT)
print(f"[{a.label}] in-dist acc:", {LAB[i]:accs[i] for i in range(3)})
print(f"[{a.label}] robustness :", {LAB[i]:robs[i] for i in range(3)})
