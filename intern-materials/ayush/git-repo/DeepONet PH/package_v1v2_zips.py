#!/usr/bin/env python
"""Stage + write data CSVs and READMEs for the two deliverable bundles:
  v1_results/  (mean-pool DeepONet, NO cross-attention)  vs flow
  v2_results/  (cross-attention DeepONet)                vs flow
Pure CPU: reads existing eval JSONs, copies existing plots/videos. Zipping is
done by the caller. Run from parent 'DeepONet PH/' dir."""
import json, csv, shutil, numpy as np
from pathlib import Path

CATS=["Camera Viewpoints","Light Conditions","Sensor Noise","Background Textures",
      "Objects Layout","Robot Initial States","Language Instructions"]

def indist(path): return json.load(open(path))["LIBERO-SPATIAL"]
def jload(path): return json.load(open(path))

def seed_in(src,pfx):  return [src[f"{pfx}_s{s}"]["average"]*100 for s in range(5) if f"{pfx}_s{s}" in src]
def seed_rob(src,pfx): return [src[f"{pfx}_s{s}"]["robustness_average"]*100 for s in range(5)
                               if f"{pfx}_s{s}" in src and src[f"{pfx}_s{s}"].get("robustness_average") is not None]
def seed_cat(src,pfx,c): return [src[f"{pfx}_s{s}"][c]["average"]*100 for s in range(5)
                               if f"{pfx}_s{s}" in src and c in src[f"{pfx}_s{s}"]]
def pertask(src,pfx):
    tab={}
    for s in range(5):
        for t,v in src.get(f"{pfx}_s{s}",{}).get("per_task",{}).items():
            tab.setdefault(int(t),[]).append(v["success_rate"]*100)
    return tab

def write_csvs(ddir, ind, pl, models):
    ddir.mkdir(parents=True, exist_ok=True)
    # summary
    with open(ddir/"summary.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["model","indist_mean","indist_std","robust_mean","robust_std","latency_ms","head_params_M","n_seeds"])
        for label,pfx,lat,par in models:
            a=seed_in(ind,pfx); r=seed_rob(pl,pfx)
            w.writerow([label,f"{np.mean(a):.1f}",f"{np.std(a):.1f}",f"{np.mean(r):.1f}",f"{np.std(r):.1f}",lat,par,len(a)])
    # per-task success
    with open(ddir/"success_per_task.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["task"]+[m[0] for m in models])
        tabs={m[1]:pertask(ind,m[1]) for m in models}
        for t in sorted(next(iter(tabs.values()))):
            w.writerow([f"task{t}"]+[f"{np.mean(tabs[m[1]][t]):.1f}" for m in models])
    # per-category robustness
    with open(ddir/"robustness_per_category.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["perturbation"]+[m[0] for m in models])
        for c in CATS:
            w.writerow([c]+[f"{np.mean(seed_cat(pl,m[1],c)):.1f}" for m in models])
    return {label:(np.mean(seed_in(ind,pfx)),np.mean(seed_rob(pl,pfx))) for label,pfx,_,_ in models}

ROOT=Path(".")
# ---------- V1 (no cross-attention) ----------
v1_ind=indist("runs/eval_indist/success_rates.json")
v1_pl =jload("runs/eval_plus/robustness_plus.json")
v1_models=[("M1 flow","m1",148.1,99.9),("M3 DeepONet-v1","m3",23.3,2.31),("M4 DeepONet+PH-v1","m4",23.3,2.31)]
d=Path("v1_results");
if d.exists(): shutil.rmtree(d)
(d/"plots").mkdir(parents=True); (d/"video").mkdir(parents=True)
for p in sorted(Path("DeepONet_Results/plots_v1").glob("*.png")): shutil.copy(p,d/"plots"/p.name)
shutil.copy("comparison_videos/flow_vs_v1__flow_WINS__indist_task08.mp4", d/"video")
shutil.copy("runs/eval_indist/success_rates.json", d/"data_indist_eval.json") if False else None
v1n=write_csvs(d/"data", v1_ind, v1_pl, v1_models)

# ---------- V2 (cross-attention) ----------
v2_ind=indist("v2/runs/eval_v2_indist/success_rates.json")
v2_pl =jload("v2/runs/eval_v2_plus/robustness_plus.json")
# flow lives in the v1 eval files (shared baseline); merge its keys so seed_* finds m1_*
for s in range(5):
    if f"m1_s{s}" in v1_ind: v2_ind[f"m1_s{s}"]=v1_ind[f"m1_s{s}"]
    if f"m1_s{s}" in v1_pl:  v2_pl[f"m1_s{s}"]=v1_pl[f"m1_s{s}"]
v2_models=[("M1 flow","m1",148.1,99.9),("M3 DeepONet-v2","m3v2",29.5,10.40),("M4 DeepONet+PH-v2","m4v2",29.5,10.40)]
d2=Path("v2_results")
if d2.exists(): shutil.rmtree(d2)
(d2/"plots").mkdir(parents=True); (d2/"video").mkdir(parents=True)
for p in sorted(Path("DeepONet_Results/plots_v2").glob("*.png")): shutil.copy(p,d2/"plots"/p.name)
shutil.copy("comparison_videos/v2_vs_flow__v2_WINS__pert_Lighting.mp4", d2/"video")
v2n=write_csvs(d2/"data", v2_ind, v2_pl, v2_models)

# ---------- READMEs ----------
def readme(path, title, sub, nums, video_desc, plotnames):
    L=[f"# {title}\n", sub, "", "## Headline numbers (LIBERO-Spatial in-dist / LIBERO-Plus robustness)\n",
       "| Model | In-dist acc | Robustness |", "|---|---|---|"]
    for k,(a,r) in nums.items(): L.append(f"| {k} | {a:.1f}% | {r:.1f}% |")
    L += ["", "## Plots (`plots/`)", ""]
    for n,desc in plotnames: L.append(f"- **{n}** — {desc}")
    L += ["", "## Video (`video/`)", "", video_desc,
          "", "## Data (`data/`)",
          "- `summary.csv` — per-model in-dist, robustness, latency, head params, #seeds",
          "- `success_per_task.csv` — in-dist success rate per LIBERO-Spatial task",
          "- `robustness_per_category.csv` — robustness per LIBERO-Plus perturbation type",
          "", "_Numbers are seed means. Flow = 5 seeds; DeepONet decks as noted in summary.csv._"]
    Path(path).write_text("\n".join(L)+"\n")

PLOTS=[("1_accuracy.png","in-distribution accuracy (success rate)"),
       ("2_robustness.png","total robustness across the 7 LIBERO-Plus perturbations"),
       ("3_latency.png","inference latency per action chunk (measured, batch 1)"),
       ("4_parameters.png","action-head parameter count (active params)"),
       ("5_success_per_task.png","success rate per LIBERO-Spatial task"),
       ("6_robustness_by_perturbation.png","robustness broken down by each LIBERO-Plus perturbation")]

readme("v1_results/README.md",
  "DeepONet-v1 (mean-pool, NO cross-attention) vs Flow-Matching",
  "The v1 action head pools the VLM prefix to a single vector (no cross-attention). "
  "It is tiny/fast (2.31M, 23ms) but trades a large chunk of accuracy and does not beat flow on robustness.",
  v1n,
  "**`flow_vs_v1__flow_WINS__indist_task08.mp4`** — side-by-side rollout on LIBERO-Spatial task 8. "
  "M1 flow (left) completes the pick-and-place (SUCCESS); DeepONet-v1 (right) fails. "
  "This is the case the mean-pool bottleneck motivated fixing in v2.",
  PLOTS)

readme("v2_results/README.md",
  "DeepONet-v2 (cross-attention) vs Flow-Matching",
  "The v2 action head uses cross-attention pooling over the full VLM prefix. It matches flow "
  "in-distribution while being markedly more robust, at ~10x fewer head params and ~5x lower latency.",
  v2n,
  "**`v2_vs_flow__v2_WINS__pert_Lighting.mp4`** — side-by-side rollout on a LIBERO-Plus *Lighting* "
  "perturbation. M1 flow (left) fails under the lighting shift; DeepONet-v2 (right) completes the task "
  "(SUCCESS). This illustrates v2's out-of-distribution robustness advantage.",
  PLOTS)

print("V1:", v1n)
print("V2:", v2n)
print("staged v1_results/ and v2_results/")
