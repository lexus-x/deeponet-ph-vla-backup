"""make_plots_v2.py — figures for the ACT V2 transfer campaign (40-task pretrain -> per-suite finetune).
Reads V2 eval outputs on the NTFS drive and writes the full figure set + summary.csv/json.
  in-dist : <BASE>/<Suite>/runs/eval_indist/success_rates.json
  Plus    : <BASE>/<Suite>/runs/eval_plus/robustness_plus.json   (LeRobot env, 7 categories)
  latency : reuses ACT/act_results/latency.json (architecture identical to V1)
Usage: python make_plots_v2.py [BASE]   (default BASE = the NTFS V2 runs dir)
"""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/media/user/C2FE578FFE577A9D/ACT_v2/runs")
OUT = BASE / "plots_all"; OUT.mkdir(parents=True, exist_ok=True)
LAT = Path("act_results/latency.json")   # same 3 architectures as V1

SUITES = ["Spatial", "Object", "Long", "Goal"]
VAR = ["act", "act_deeponet", "act_deeponet_ph"]
LAB = {"act": "ACT", "act_deeponet": "ACT+DeepONet", "act_deeponet_ph": "ACT+DeepONet+PH"}
COL = {"act": "#4C72B0", "act_deeponet": "#DD8452", "act_deeponet_ph": "#55A868"}
CATS = ["Camera Viewpoints", "Light Conditions", "Sensor Noise", "Background Textures",
        "Objects Layout", "Robot Initial States", "Language Instructions"]
CAT_SHORT = ["Camera", "Light", "SensorNoise", "Background", "ObjLayout", "RobotInit", "Language"]

def load(p):
    try: return json.load(open(p))
    except Exception: return None

indist, plus = {}, {}
for s in SUITES:
    d = load(BASE / s / "runs/eval_indist/success_rates.json")
    if d:
        skey = [k for k in d if not k.startswith("_")][0]
        indist[s] = {}
        for v in VAR:
            mv = d[skey].get(v)
            if mv and "per_task" in mv:
                rates = [t["success_rate"] for t in mv["per_task"].values()]
                seedmat = [t.get("per_seed", []) for t in mv["per_task"].values()]
                ns = min((len(x) for x in seedmat), default=0)
                pso = [np.mean([r[i] for r in seedmat]) for i in range(ns)] if ns else []
                indist[s][v] = {"overall": float(np.mean(rates))*100,
                                "std": float(np.std(pso))*100 if pso else 0.0,
                                "per_task": {k: (t["task"], t["success_rate"]*100) for k, t in mv["per_task"].items()}}
    d = load(BASE / s / "runs/eval_plus/robustness_plus.json")
    if d:
        plus[s] = {}
        for v in VAR:
            mv = d.get(v)
            if mv:
                plus[s][v] = {"overall": mv.get("robustness_average", float("nan"))*100,
                              "cats": {c: mv.get(c, {}).get("average", float("nan"))*100 for c in CATS}}

lat = load(LAT)
PARAM = {"act": 88.284, "act_deeponet": 60.712, "act_deeponet_ph": 60.712}

def avg(table, key):
    out = {}
    for v in VAR:
        vals = [table[s][v][key] for s in table if v in table[s] and not math.isnan(table[s][v][key])]
        out[v] = float(np.mean(vals)) if vals else float("nan")
    return out

def lbl(ax, bars, fmt="{:.1f}", fs=8):
    for b in bars:
        h = b.get_height()
        if not math.isnan(h): ax.text(b.get_x()+b.get_width()/2, h, fmt.format(h), ha="center", va="bottom", fontsize=fs)

def grouped(ax, groups, vals, err=None, ylabel="", title="", ylim=None, w=0.26, rot=0):
    x = np.arange(len(groups)); n = len(VAR)
    for i, v in enumerate(VAR):
        bars = ax.bar(x+(i-(n-1)/2)*w, vals[v], w, label=LAB[v], color=COL[v],
                      yerr=(err[v] if err else None), capsize=3, edgecolor="black", linewidth=0.4)
        lbl(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=rot, ha=("center" if rot==0 else "right"))
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True); ax.legend(fontsize=8)

saved = []
def save(fig, n): p = OUT/n; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(n)

if not indist:
    print("No V2 eval results found yet under", BASE); sys.exit(0)

# in-dist overall
vv = {v: [indist.get(s, {}).get(v, {}).get("overall", float("nan")) for s in SUITES] for v in VAR}
ee = {v: [indist.get(s, {}).get(v, {}).get("std", 0.0) for s in SUITES] for v in VAR}
ai = avg(indist, "overall")
for v in VAR: vv[v].append(ai[v]); ee[v].append(0.0)
fig, ax = plt.subplots(figsize=(11, 5))
grouped(ax, SUITES+["Average"], vv, ee, "Success rate (%)",
        "V2 in-distribution success (40-task pretrain -> per-suite finetune; 3-seed std)", ylim=(0,100))
save(fig, "03_indist_overall.png")

# plus overall
if plus:
    vv = {v: [plus.get(s, {}).get(v, {}).get("overall", float("nan")) for s in SUITES] for v in VAR}
    ap = avg(plus, "overall")
    for v in VAR: vv[v].append(ap[v])
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped(ax, SUITES+["Average"], vv, None, "Robustness (%)",
            "V2 out-of-distribution: LIBERO-Plus (7 perturbations)", ylim=(0,80))
    save(fig, "04_plus_overall.png")

    # per-category avg
    catavg = {v: [] for v in VAR}
    for v in VAR:
        for c in CATS:
            cv = [plus[s][v]["cats"][c] for s in SUITES if v in plus.get(s, {}) and not math.isnan(plus[s][v]["cats"][c])]
            catavg[v].append(float(np.mean(cv)) if cv else float("nan"))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    grouped(ax, CAT_SHORT, catavg, None, "Robustness (%)", "V2 LIBERO-Plus per-perturbation (avg over suites)", ylim=(0,90), rot=25)
    save(fig, "07_perturbation_by_category_avg.png")

    # heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    for ax, v in zip(axes, VAR):
        M = np.array([[plus.get(s, {}).get(v, {}).get("cats", {}).get(c, float("nan")) for c in CATS] for s in SUITES])
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CAT_SHORT, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(SUITES))); ax.set_yticklabels(SUITES); ax.set_title(LAB[v])
        for i in range(len(SUITES)):
            for j in range(len(CATS)):
                if not math.isnan(M[i,j]): ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("V2 LIBERO-Plus heatmap: suite x perturbation", fontsize=13)
    save(fig, "09_perturbation_heatmaps.png")

# per-task grid
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, s in zip(axes.flat, SUITES):
    if s not in indist or "act" not in indist[s]: ax.set_title(f"{s} (no data)"); continue
    tids = sorted(indist[s]["act"]["per_task"].keys(), key=lambda k: int(k))
    vt = {v: [indist[s].get(v, {}).get("per_task", {}).get(t, ("",float("nan")))[1] for t in tids] for v in VAR}
    grouped(ax, [f"T{t}" for t in tids], vt, None, "Success (%)", f"{s} per-task (V2)", ylim=(0,105))
fig.suptitle("V2 per-task in-distribution success", fontsize=13)
save(fig, "06_per_task_indist.png")

# master summary
ai = avg(indist, "overall"); ap = avg(plus, "overall") if plus else {v: float("nan") for v in VAR}
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax = axes[0,0]; b = ax.bar([LAB[v] for v in VAR], [PARAM[v] for v in VAR], color=[COL[v] for v in VAR], edgecolor="black", lw=0.4); lbl(ax, b, "{:.1f}M", 9); ax.set_title("Parameters (M)"); ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
ax = axes[0,1]
if lat: b = ax.bar([LAB[v] for v in VAR], [lat[v]["plan_forward"]["mean_ms"] for v in VAR], color=[COL[v] for v in VAR], edgecolor="black", lw=0.4); lbl(ax, b, "{:.1f}", 9); ax.set_title("Planning latency (ms)"); ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
ax = axes[1,0]; b = ax.bar([LAB[v] for v in VAR], [ai[v] for v in VAR], color=[COL[v] for v in VAR], edgecolor="black", lw=0.4); lbl(ax, b); ax.set_title("In-dist success (avg, %)"); ax.set_ylim(0,100); ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
ax = axes[1,1]; b = ax.bar([LAB[v] for v in VAR], [ap[v] for v in VAR], color=[COL[v] for v in VAR], edgecolor="black", lw=0.4); lbl(ax, b); ax.set_title("LIBERO-Plus OOD (avg, %)"); ax.set_ylim(0,80); ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
fig.suptitle("ACT V2 summary (40-task pretrain -> per-suite finetune)", fontsize=13)
save(fig, "12_master_summary.png")

# summary csv/json
rows = ["metric,suite,ACT,ACT+DeepONet,ACT+DeepONet+PH"]
for s in SUITES:
    if s in indist: rows.append(f"in_dist,{s}," + ",".join(f"{indist[s].get(v,{}).get('overall',float('nan')):.1f}" for v in VAR))
rows.append("in_dist,Average," + ",".join(f"{ai[v]:.1f}" for v in VAR))
for s in SUITES:
    if s in plus: rows.append(f"plus,{s}," + ",".join(f"{plus[s].get(v,{}).get('overall',float('nan')):.1f}" for v in VAR))
if plus: rows.append("plus,Average," + ",".join(f"{ap[v]:.1f}" for v in VAR))
(OUT/"summary.csv").write_text("\n".join(rows)+"\n")
(OUT/"summary.json").write_text(json.dumps({"in_dist": ai, "plus": ap}, indent=2))
print(f"Saved {len(saved)} figures + summary to {OUT}:")
for n in saved: print("  ", n)
