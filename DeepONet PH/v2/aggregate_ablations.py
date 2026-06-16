#!/usr/bin/env python
"""Ablation table: full DeepONet-v2 (reference, 5 seeds) vs each component removed
(p64 / noF / 1blk, 2 seeds). In-dist + robustness, mean±std. Saves CSV + bar plot
into ../DeepONet_Results/. Run from v2/."""
import json, csv, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CATS=["Camera Viewpoints","Light Conditions","Sensor Noise","Background Textures",
      "Objects Layout","Robot Initial States","Language Instructions"]
def jload(p): p=Path(p); return json.loads(p.read_text()) if p.exists() else {}

full_in=jload("runs/eval_v2_indist/success_rates.json").get("LIBERO-SPATIAL",{})
full_pl=jload("runs/eval_v2_plus/robustness_plus.json")
abl_in=jload("runs/eval_abl_indist/success_rates.json").get("LIBERO-SPATIAL",{})
abl_pl=jload("runs/eval_abl_plus/robustness_plus.json")

def indist_avg(src,prefix,seeds):
    return [src[f"{prefix}_s{s}"]["average"]*100 for s in seeds
            if f"{prefix}_s{s}" in src and src[f"{prefix}_s{s}"].get("average") is not None]
def robust_avg(src,prefix,seeds):
    return [src[f"{prefix}_s{s}"]["robustness_average"]*100 for s in seeds
            if f"{prefix}_s{s}" in src and src[f"{prefix}_s{s}"].get("robustness_average") is not None]

rows=[]
# reference: full v2 (5 seeds)
configs=[("Full DeepONet-v2 (p256, 3blk, Fourier)", full_in, full_pl, "m3v2", range(5)),
         ("(-) basis p: 256->64",                   abl_in, abl_pl, "abl_p64", range(3)),
         ("(-) Fourier-tau (linear only)",          abl_in, abl_pl, "abl_noF", range(3)),
         ("(-) cross-attn blocks: 3->1",            abl_in, abl_pl, "abl_1blk", range(3)),
         ("Regression head (NO operator, same ctx)", abl_in, abl_pl, "reg", range(3))]
for label, si, sp, pfx, seeds in configs:
    a=indist_avg(si,pfx,seeds); r=robust_avg(sp,pfx,seeds)
    am=np.mean(a) if a else float("nan"); asd=np.std(a) if a else 0
    rm=np.mean(r) if r else float("nan"); rsd=np.std(r) if r else 0
    rows.append((label,am,asd,rm,rsd,len(a)))

print("\n=== DeepONet-v2 design ablations ===")
print(f"{'config':42s} {'in-dist':>14s} {'robustness':>14s}  n")
for label,am,asd,rm,rsd,n in rows:
    print(f"{label:42s} {am:6.1f}±{asd:4.1f}    {rm:6.1f}±{rsd:4.1f}   {n}")

import shutil
out=Path("../Ablation_Results"); out.mkdir(parents=True,exist_ok=True)
with open(out/"ablations.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["config","indist_mean","indist_std","robust_mean","robust_std","n_seeds"])
    for r in rows: w.writerow([r[0],f"{r[1]:.1f}",f"{r[2]:.1f}",f"{r[3]:.1f}",f"{r[4]:.1f}",r[5]])
# copy raw ablation eval JSONs into the separate folder
for src,dst in [("runs/eval_abl_indist/success_rates.json","ablation_indist.json"),
                ("runs/eval_abl_plus/robustness_plus.json","ablation_robustness.json")]:
    if Path(src).exists(): shutil.copy(src, out/dst)

# grouped bar plot
labels=[r[0].replace("Full DeepONet-v2 (p256, 3blk, Fourier)","Full v2").replace("(-) ","–").replace(": 256->64","").replace(" (linear only)","").replace(": 3->1","") for r in rows]
x=np.arange(len(rows)); w=0.38
fig,ax=plt.subplots(figsize=(11,5.5))
ax.bar(x-w/2,[r[1] for r in rows],w,yerr=[r[2] for r in rows],capsize=5,label="in-dist acc",color="#2E86DE")
ax.bar(x+w/2,[r[3] for r in rows],w,yerr=[r[4] for r in rows],capsize=5,label="robustness",color="#EE5253")
for i,r in enumerate(rows):
    ax.text(i-w/2,r[1]+1,f"{r[1]:.0f}",ha="center",fontsize=8,fontweight="bold")
    ax.text(i+w/2,r[3]+1,f"{r[3]:.0f}",ha="center",fontsize=8,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8,rotation=10); ax.set_ylim(0,100)
ax.set_ylabel("success rate (%)"); ax.legend()
ax.set_title("DeepONet-v2 design ablations (in-dist 5-seed ref; ablations 3-seed)",fontweight="bold")
ax.grid(alpha=.25,axis="y"); fig.tight_layout()
fig.savefig(out/"ablations.png",dpi=150); plt.close(fig)
print("saved ->", out, "(ablations.csv, ablations.png, raw JSONs)")
