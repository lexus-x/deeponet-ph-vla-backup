"""Comprehensive plotting for the ACT campaign: ACT vs ACT+DeepONet vs ACT+DeepONet+PH.
Generates the full figure set into act_results/plots_all/.
Data sources (canonical):
  - in-dist  : act_results/{suite}/runs/eval_rerun_indist/success_rates.json  (10 eps x 3 seeds)
  - LIBERO-Plus (OOD, 7 perturbations): act_results/{suite}/runs/eval_lerobot_full/robustness_plus.json
  - latency  : act_results/latency.json  (this GPU, bf16, batch=1)
  - params   : run_config.json + architecture.md component breakdown
"""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("act_results")
OUT = R / "plots_all"; OUT.mkdir(parents=True, exist_ok=True)

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

# ---------------------------------------------------------------- gather data
indist = {}        # indist[suite][var] = {overall, std, per_task:{id:(name,rate)}, per_seed_overall:[..]}
for s in SUITES:
    d = load(R / s / "runs/eval_rerun_indist/success_rates.json")
    if not d: continue
    skey = [k for k in d if not k.startswith("_")][0]
    indist[s] = {}
    for v in VAR:
        mv = d[skey].get(v)
        if not mv or "per_task" not in mv: continue
        tasks = mv["per_task"]
        rates = [t["success_rate"] for t in tasks.values()]
        # 3-seed overall: mean across tasks per seed
        seedmat = [t.get("per_seed", []) for t in tasks.values()]
        nseed = min((len(x) for x in seedmat), default=0)
        per_seed_overall = [np.mean([row[i] for row in seedmat]) for i in range(nseed)] if nseed else []
        indist[s][v] = {
            "overall": float(np.mean(rates)) * 100,
            "std": float(np.std(per_seed_overall)) * 100 if per_seed_overall else 0.0,
            "per_task": {k: (t["task"], t["success_rate"] * 100) for k, t in tasks.items()},
        }

plus = {}          # plus[suite][var] = {overall, cats:{cat:val}}
for s in SUITES:
    d = load(R / s / "runs/eval_lerobot_full/robustness_plus.json")
    if not d: continue
    plus[s] = {}
    for v in VAR:
        mv = d.get(v)
        if not mv: continue
        plus[s][v] = {
            "overall": mv.get("robustness_average", float("nan")) * 100,
            "cats": {c: mv.get(c, {}).get("average", float("nan")) * 100 for c in CATS},
        }

lat = load(R / "latency.json")
PARAM_TOTAL = {"act": 88.284, "act_deeponet": 60.712, "act_deeponet_ph": 60.712}
PARAM_BACKBONE = 11.2
PARAM_HEAD = {"act": 37.8, "act_deeponet": 10.2, "act_deeponet_ph": 10.2}

def avg_over_suites(table, key):
    """mean across suites of table[suite][var][key]"""
    out = {}
    for v in VAR:
        vals = [table[s][v][key] for s in table if v in table[s] and not math.isnan(table[s][v][key])]
        out[v] = float(np.mean(vals)) if vals else float("nan")
    return out

def barlabels(ax, bars, fmt="{:.1f}", fs=8):
    for b in bars:
        h = b.get_height()
        if math.isnan(h): continue
        ax.text(b.get_x() + b.get_width()/2, h, fmt.format(h), ha="center", va="bottom", fontsize=fs)

def grouped(ax, groups, series_vals, series_err=None, ylabel="", title="", ylim=None, width=0.26, rot=0):
    x = np.arange(len(groups)); n = len(VAR)
    for i, v in enumerate(VAR):
        off = (i - (n-1)/2) * width
        err = series_err[v] if series_err else None
        bars = ax.bar(x + off, series_vals[v], width, label=LAB[v], color=COL[v],
                      yerr=err, capsize=3, edgecolor="black", linewidth=0.4)
        barlabels(ax, bars)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=rot, ha="center" if rot==0 else "right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=8)

saved = []
def save(fig, name):
    p = OUT / name; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(name)

# ======================================================== 1. PARAMETERS
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
bars = ax.bar([LAB[v] for v in VAR], [PARAM_TOTAL[v] for v in VAR], color=[COL[v] for v in VAR],
              edgecolor="black", linewidth=0.5)
barlabels(ax, bars, "{:.1f}M", 10)
ax.set_ylabel("Total parameters (M)"); ax.set_title("Total parameter count")
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True); ax.tick_params(axis="x", rotation=12)
ax = axes[1]
xb = np.arange(len(VAR))
b1 = ax.bar(xb, [PARAM_BACKBONE]*len(VAR), 0.55, label="ResNet-18 backbone (shared)", color="#BBBBBB", edgecolor="black", lw=0.4)
b2 = ax.bar(xb, [PARAM_HEAD[v] for v in VAR], 0.55, bottom=[PARAM_BACKBONE]*len(VAR),
            label="Action head", color=[COL[v] for v in VAR], edgecolor="black", lw=0.4)
for i, v in enumerate(VAR):
    ax.text(i, PARAM_BACKBONE/2, f"{PARAM_BACKBONE:.1f}M", ha="center", va="center", fontsize=8)
    ax.text(i, PARAM_BACKBONE + PARAM_HEAD[v]/2, f"head\n{PARAM_HEAD[v]:.1f}M", ha="center", va="center", fontsize=8)
ax.set_xticks(xb); ax.set_xticklabels([LAB[v] for v in VAR], rotation=12)
ax.set_ylabel("Parameters (M)"); ax.set_title("Backbone vs action-head breakdown")
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True); ax.legend(fontsize=8)
fig.suptitle("Model size — DeepONet head is 3.7x smaller (37.8M -> 10.2M), 31% fewer total params", fontsize=11)
save(fig, "01_parameters.png")

# ======================================================== 2. LATENCY
if lat:
    plan = {v: lat[v]["plan_forward"]["mean_ms"] for v in VAR}
    plan_e = {v: lat[v]["plan_forward"]["std_ms"] for v in VAR}
    amort = {v: lat[v]["amortized_replan5"]["mean_ms"] for v in VAR}
    hz = {v: lat[v]["control_freq_hz_amort"] for v in VAR}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, data, err, ttl, yl in [
        (axes[0], plan, plan_e, "Planning forward pass\n(full action-chunk prediction)", "latency (ms)"),
        (axes[1], amort, None, "Amortized per control step\n(receding horizon, replan=5)", "latency (ms)"),
        (axes[2], hz, None, "Control frequency\n(amortized, replan=5)", "Hz")]:
        bars = ax.bar([LAB[v] for v in VAR], [data[v] for v in VAR], color=[COL[v] for v in VAR],
                      yerr=[err[v] for v in VAR] if err else None, capsize=4, edgecolor="black", lw=0.5)
        barlabels(ax, bars, "{:.1f}", 9)
        ax.set_ylabel(yl); ax.set_title(ttl); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle(f"Inference latency — {lat['_meta']['gpu']}, bf16, batch=1", fontsize=11)
    save(fig, "02_latency.png")

# ======================================================== 3. IN-DIST overall
vals = {v: [indist[s][v]["overall"] for s in SUITES] for v in VAR}
errs = {v: [indist[s][v]["std"] for s in SUITES] for v in VAR}
avg = avg_over_suites(indist, "overall")
for v in VAR: vals[v].append(avg[v]); errs[v].append(0.0)
fig, ax = plt.subplots(figsize=(11, 5))
grouped(ax, SUITES + ["Average"], vals, errs, "Success rate (%)",
        "In-distribution success rate (10 episodes x 3 seeds; error bars = 3-seed std)", ylim=(0, 100))
save(fig, "03_indist_overall.png")

# ======================================================== 4. PLUS overall (OOD)
vals = {v: [plus[s][v]["overall"] for s in SUITES] for v in VAR}
avg = avg_over_suites(plus, "overall")
for v in VAR: vals[v].append(avg[v])
fig, ax = plt.subplots(figsize=(11, 5))
grouped(ax, SUITES + ["Average"], vals, None, "Robustness success rate (%)",
        "Out-of-distribution: LIBERO-Plus overall robustness (7 perturbations x 12 tasks)", ylim=(0, 80))
save(fig, "04_plus_overall.png")

# ======================================================== 5. IN-DIST vs OOD (drop)
ai = avg_over_suites(indist, "overall"); ap = avg_over_suites(plus, "overall")
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(VAR)); w = 0.36
b1 = ax.bar(x - w/2, [ai[v] for v in VAR], w, label="In-distribution", color="#4C72B0", edgecolor="black", lw=0.4)
b2 = ax.bar(x + w/2, [ap[v] for v in VAR], w, label="LIBERO-Plus (OOD)", color="#C44E52", edgecolor="black", lw=0.4)
barlabels(ax, b1); barlabels(ax, b2)
for i, v in enumerate(VAR):
    drop = ai[v] - ap[v]
    ax.annotate(f"-{drop:.1f}", (i, max(ai[v], ap[v]) + 3), ha="center", fontsize=9, color="black")
ax.set_xticks(x); ax.set_xticklabels([LAB[v] for v in VAR], rotation=10)
ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 100)
ax.set_title("In-distribution vs OOD (averaged over 4 suites) — robustness drop annotated")
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True); ax.legend()
save(fig, "05_indist_vs_plus_drop.png")

# ======================================================== 6. PER-TASK accuracy (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, s in zip(axes.flat, SUITES):
    tids = sorted(indist[s]["act"]["per_task"].keys(), key=lambda k: int(k))
    groups = [f"T{t}" for t in tids]
    vv = {v: [indist[s][v]["per_task"][t][1] for t in tids] for v in VAR}
    grouped(ax, groups, vv, None, "Success rate (%)", f"{s} — per-task in-distribution", ylim=(0, 105))
fig.suptitle("Per-task in-distribution success rate (3-seed mean)", fontsize=13)
save(fig, "06_per_task_indist.png")

# ======================================================== 7. PER-PERTURBATION averaged over suites
catavg = {v: [] for v in VAR}
for v in VAR:
    for c in CATS:
        vals_c = [plus[s][v]["cats"][c] for s in SUITES if v in plus[s] and not math.isnan(plus[s][v]["cats"][c])]
        catavg[v].append(float(np.mean(vals_c)) if vals_c else float("nan"))
fig, ax = plt.subplots(figsize=(13, 5.5))
grouped(ax, CAT_SHORT, catavg, None, "Robustness success rate (%)",
        "LIBERO-Plus per-perturbation robustness (averaged over 4 suites)", ylim=(0, 90), rot=25)
save(fig, "07_perturbation_by_category_avg.png")

# ======================================================== 8. PER-PERTURBATION per suite (2x2)
fig, axes = plt.subplots(2, 2, figsize=(18, 11))
for ax, s in zip(axes.flat, SUITES):
    vv = {v: [plus[s][v]["cats"][c] for c in CATS] for v in VAR}
    grouped(ax, CAT_SHORT, vv, None, "Robustness (%)", f"{s} — per-perturbation", ylim=(0, 105), rot=30)
fig.suptitle("LIBERO-Plus: per-perturbation robustness by suite (7 categories)", fontsize=13)
save(fig, "08_perturbation_per_suite.png")

# ======================================================== 9. HEATMAPS (suite x category) per variant
fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
for ax, v in zip(axes, VAR):
    M = np.array([[plus[s][v]["cats"][c] for c in CATS] for s in SUITES])
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CAT_SHORT, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(SUITES))); ax.set_yticklabels(SUITES)
    ax.set_title(LAB[v])
    for i in range(len(SUITES)):
        for j in range(len(CATS)):
            ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=8,
                    color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="success %")
fig.suptitle("LIBERO-Plus robustness heatmap: suite x perturbation, per variant", fontsize=13)
save(fig, "09_perturbation_heatmaps.png")

# ======================================================== 10. RADAR (per-perturbation, avg over suites)
ang = np.linspace(0, 2*np.pi, len(CATS), endpoint=False).tolist(); ang += ang[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for v in VAR:
    data = catavg[v] + catavg[v][:1]
    ax.plot(ang, data, label=LAB[v], color=COL[v], linewidth=2)
    ax.fill(ang, data, color=COL[v], alpha=0.12)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(CAT_SHORT, fontsize=9)
ax.set_ylim(0, 80); ax.set_title("Perturbation robustness profile (avg over suites)", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
save(fig, "10_perturbation_radar.png")

# ======================================================== 11. EFFICIENCY (Pareto)
ind_avg = avg_over_suites(indist, "overall"); plus_avg = avg_over_suites(plus, "overall")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
if lat:
    planL = {v: lat[v]["plan_forward"]["mean_ms"] for v in VAR}
    ax = axes[0]
    for v in VAR:
        ax.scatter(planL[v], ind_avg[v], s=PARAM_TOTAL[v]*12, color=COL[v], edgecolor="black", zorder=3, label=LAB[v])
        ax.annotate(f"{LAB[v]}\n{ind_avg[v]:.1f}%, {planL[v]:.1f}ms", (planL[v], ind_avg[v]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.set_xlabel("Planning latency (ms, lower better)"); ax.set_ylabel("In-dist success (%)")
    ax.set_title("Accuracy vs latency (bubble = params)"); ax.grid(alpha=0.3)
ax = axes[1]
for v in VAR:
    ax.scatter(PARAM_TOTAL[v], ind_avg[v], s=160, color=COL[v], edgecolor="black", zorder=3, marker="o", label=f"{LAB[v]} in-dist")
    ax.scatter(PARAM_TOTAL[v], plus_avg[v], s=160, color=COL[v], edgecolor="black", zorder=3, marker="^")
    ax.annotate(f"{ind_avg[v]:.1f}", (PARAM_TOTAL[v], ind_avg[v]), textcoords="offset points", xytext=(8, 0), fontsize=8)
    ax.annotate(f"{plus_avg[v]:.1f}", (PARAM_TOTAL[v], plus_avg[v]), textcoords="offset points", xytext=(8, 0), fontsize=8)
ax.set_xlabel("Total parameters (M)"); ax.set_ylabel("Success (%)")
ax.set_title("Accuracy vs params  (o = in-dist, ^ = OOD)"); ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="center right")
fig.suptitle("Efficiency trade-offs", fontsize=12)
save(fig, "11_efficiency_pareto.png")

# ======================================================== 12. MASTER summary (2x2)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# params
ax = axes[0,0]; bars = ax.bar([LAB[v] for v in VAR], [PARAM_TOTAL[v] for v in VAR], color=[COL[v] for v in VAR], edgecolor="black", lw=0.4)
barlabels(ax, bars, "{:.1f}M", 9); ax.set_title("Parameters (M)"); ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
# latency
ax = axes[0,1]
if lat:
    bars = ax.bar([LAB[v] for v in VAR], [lat[v]["plan_forward"]["mean_ms"] for v in VAR], color=[COL[v] for v in VAR], edgecolor="black", lw=0.4)
    barlabels(ax, bars, "{:.1f}", 9); ax.set_title("Planning latency (ms)"); ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
# indist avg
ax = axes[1,0]; bars = ax.bar([LAB[v] for v in VAR], [ind_avg[v] for v in VAR], color=[COL[v] for v in VAR], edgecolor="black", lw=0.4)
barlabels(ax, bars); ax.set_title("In-dist success (avg 4 suites, %)"); ax.set_ylim(0,100); ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
# plus avg
ax = axes[1,1]; bars = ax.bar([LAB[v] for v in VAR], [plus_avg[v] for v in VAR], color=[COL[v] for v in VAR], edgecolor="black", lw=0.4)
barlabels(ax, bars); ax.set_title("LIBERO-Plus OOD (avg 4 suites, %)"); ax.set_ylim(0,80); ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
fig.suptitle("ACT campaign summary — ACT vs ACT+DeepONet vs ACT+DeepONet+PH", fontsize=13)
save(fig, "12_master_summary.png")

# ======================================================== dump a tidy CSV + json
summary = {"in_dist": {}, "libero_plus": {}, "params_M": PARAM_TOTAL,
           "latency": {v: lat[v] for v in VAR} if lat else {},
           "averages": {"in_dist": ind_avg, "libero_plus": plus_avg}}
for s in SUITES:
    summary["in_dist"][s] = {v: round(indist[s][v]["overall"], 1) for v in VAR}
    summary["libero_plus"][s] = {v: {"overall": round(plus[s][v]["overall"], 1),
                                     "by_cat": {c: round(plus[s][v]["cats"][c], 1) for c in CATS}} for v in VAR}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

rows = ["metric,suite/category,ACT,ACT+DeepONet,ACT+DeepONet+PH"]
for s in SUITES:
    rows.append(f"in_dist,{s}," + ",".join(f"{indist[s][v]['overall']:.1f}" for v in VAR))
rows.append("in_dist,Average," + ",".join(f"{ind_avg[v]:.1f}" for v in VAR))
for s in SUITES:
    rows.append(f"plus_overall,{s}," + ",".join(f"{plus[s][v]['overall']:.1f}" for v in VAR))
rows.append("plus_overall,Average," + ",".join(f"{plus_avg[v]:.1f}" for v in VAR))
for ci, c in enumerate(CATS):
    rows.append(f"plus_cat_avg,{c}," + ",".join(f"{catavg[v][ci]:.1f}" for v in VAR))
rows.append("params_M,-," + ",".join(f"{PARAM_TOTAL[v]:.1f}" for v in VAR))
if lat:
    rows.append("plan_latency_ms,-," + ",".join(f"{lat[v]['plan_forward']['mean_ms']:.2f}" for v in VAR))
    rows.append("amortized_ms,-," + ",".join(f"{lat[v]['amortized_replan5']['mean_ms']:.2f}" for v in VAR))
(OUT / "summary.csv").write_text("\n".join(rows) + "\n")

print(f"Saved {len(saved)} figures to {OUT}/:")
for n in saved: print("  ", n)
print("Also wrote summary.json and summary.csv")
