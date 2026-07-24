# DeepONet contribution restart — research memo (2026-07-24)

**Constraint:** No backing down. Prior bets (pin / ens / short POD) **failed on SR**. Contribution must pivot to gaps that are still *open, ours, and publishable*.

## What the failed campaign actually taught us

| Result | Scientific reading |
|--------|-------------------|
| Ens 55.5% < m3 60% | Temporal averaging **hurts** when replans jump modes ([BID](https://bid-robot.github.io/) warns EMA can be detrimental across modes) |
| Pin ~5% | Exact boundary continuity ≠ task success; Exp A already predicted +tracking RMSE / drag |
| POD 58% vs M3 77% | **Confounded**: POD trained **8.3K**; Fourier M3 is **30K**. Not a fair trunk ablation yet |
| Spatial tie / Long flow+12 | Paper’s honest story is **robustness + efficiency**, not in-dist / Long win ([CORRECTIONS_2026-06-25](../intern-materials/ayush/git-repo/DeepONet%20PH/CORRECTIONS_2026-06-25.md)) |

→ Stop chasing Long in-dist with pin/ens. Chase **the claim the paper already stakes**: multi-suite robustness + mechanism science.

---

## Ranked bets (EV for co-lead)

### A — ★★★★★ Multi-suite LIBERO-Plus (Object / Long / Goal) — **PRIMARY**

**Why this is the paper hole:** Ayush’s “>2× more robust” is **Spatial-only**. The repo itself flags: *no Plus numbers for Object / Long / Goal* ([CORRECTIONS §5](../intern-materials/ayush/git-repo/DeepONet%20PH/CORRECTIONS_2026-06-25.md)).

**Why literature backs urgency:** [LIBERO-Plus](https://sylvestf.github.io/LIBERO-plus/) / [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf) is now the standard multi-factor robustness suite (7 dims, all LIBERO suites). Completing M3 vs flow Plus on Object/Long/Goal is both **required for an honest paper** and a potential **headline win** if Spatial’s robustness advantage generalizes.

**Assets ready on blackwell:** Plus package installed; Object/Long/Spatial m3+flow @30K present; GPU idle.

**Success bar:** M3 Plus avg ≫ flow on ≥2 suites, or clear per-perturbation story even if mixed.

**Credit type:** Named multi-suite completion + tables — meets authorship “multi-suite Plus completion.”

---

### B — ★★★★ Fair POD / trunk ablations (matched budget)

Retrain POD (and optionally p64 / soft-orthogonal Fourier) at **30K Spatial** = M3 budget. Current −19 pp is not interpretable.

Optional hybrid: **fixed POD basis + learned residual trunk** (still “ours”).

**Credit type:** Named method + ablation (authorship bar #2) — only if matched-budget POD ≥ or near Fourier, or cleanly explains when data-driven trunk helps.

---

### C — ★★★★ Mechanism paper: Long / chunking diagnosis (even if SR negative)

Publishable without beating flow:

1. Replan-1 vs r5 (already: −15.5 pp)  
2. Ens/pin fail on Long + boundary-jump metric on success vs failure (Exp A prediction)  
3. Tie to [chunk-boundary artifact](https://arxiv.org/html/2603.11642) + [BID](https://arxiv.org/html/2408.17355v2) — DeepONet is **deterministic** (no multi-sample BID); propose **DeepONet-native BID**: query τ-grid variants / dropout branch / noise on trunk coords and pick by backward coherence  

**Credit type:** Analysis + new decoding method for operator heads (not flow/diffusion sampling).

---

### D — ★★★ PH anneal / drop + PCA-3 surrogate (negative → protocol)

Already partially done in memo; GPU confirm λ_PH→0 late restores Plus. Strengthens “we fixed the training recipe” contribution; not lead invention alone.

---

### E — ★★ Efficiency + claim rewrite vs NIAF / Spline Policy

Narrow novelty: **VLM-conditioned branch × (Fourier or POD) trunk** for SmolVLA, with efficiency (~9.6× / ~5×) and Spatial Plus. Doesn’t need Long win; needs clean related-work + ablations.

---

## Explicitly deprioritize (for now)

| Idea | Why |
|------|-----|
| More pin/ens hyperparam sweeps on Long | Theory + data say averaging across modes is wrong tool |
| Claiming sole DeepONet invention | Ayush pre–Jul 15 |
| POD as headline with 8.3K recipe | Invalid comparison |
| Waiting on a100 | Busy with unrelated bridge pretrain; **blackwell is free** |

---

## Recommended 72h plan (execute on blackwell)

1. **Now:** Launch LIBERO-Plus M3 vs flow — Spatial (sanity vs Ayush table) → Object → Long (Goal if ckpt exists).  
2. **Parallel analysis:** Boundary-jump logs on Long success/fail episodes (no new train).  
3. **If Plus Object/Long moves:** draft paper table + co-lead framing.  
4. **Else:** start matched-budget POD 30K + DeepONet-BID prototype.

## Authorship bar reminder

Need ≥2 of: (1) closed-loop win from our work, (2) named method + ablation, (3) honest framing.  
**A alone can deliver (1)+(3). A+B or A+C delivers all three.**
