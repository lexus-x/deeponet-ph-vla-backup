# Corrections & statistical audit (2026-06-25)

This note documents corrections to overstated claims in the README/paper, each **verified
against the raw result JSONs** in this repo. Backing data is committed alongside the headline
plots in [`runs/report_final/`](runs/report_final/):
`seed_level_spatial_8p3k.csv`, `significance_tests.csv`, `budget_comparison.csv`.

Model legend: **M1 = flow** (SmolVLA baseline) · **M3 = DeepONet** · **M4 = DeepONet + PH**.
All statistics are the LIBERO-Spatial 5-seed set unless noted (the only multi-seed regime).

---

### 1. In-distribution is a statistical TIE, not a "match"/win
- M3 DeepONet-v2 **80.30 ± 2.42 %** vs M1 flow **79.40 ± 1.39 %** → Δ = **+0.90 pp**, **t-p = 0.605** (n=50).
- This is **not significant**: in-distribution accuracy is a tie. "Matches in-distribution" should be stated as
  "is statistically indistinguishable in-distribution (p = 0.61)".

### 2. The operator does NOT drive in-distribution accuracy
- A **regression head with NO operator merge** (same cross-attention context) scores **83.0 %** in-dist —
  *higher* than the full DeepONet (80.3 %). The "(−) Fourier" (82.7 %) and "(−) p256→64" (81.7 %) ablations are
  also ≥ the full operator in-dist.
- → In-distribution accuracy comes from the **cross-attention pooler / context**, not from the operator merge.
  The operator's measurable contribution is **robustness** (regression head 32.7 % vs operator 38.5 %).

### 3. PH loss HURTS robustness (significantly)
- M4 (DeepONet + PH) robustness **32.57 %** vs M3 (DeepONet) **38.48 %** → Δ = **−5.90 pp**, **t-p = 0.032** (n=35;
  Wilcoxon p = 0.056, marginal).
- PH's in-distribution "gain" (+1.20 pp, M4 vs M3) is **not significant** (t-p = 0.51).
- → PH is **not** recommended: it does not significantly help in-dist and significantly degrades robustness.
  The headline operator model is **M3 (PH off)**.

### 4. At a FIXED budget the flow baseline wins the in-dist whole-suite average
- The README's "M3 edges out flow (81.88 vs 81.00)" uses **per-suite best-budget selection** — specifically
  **Object @ 15K = 94.0** for M3 (M3 over-performs at 15K on Object) while flow uses its weaker Object@15K = 84.5.
- At a **consistent 30K budget** (Object @ 30K for all; Goal only exists at 15K), the ranking flips:

  | selection | M1 flow | M3 DeepONet | M4 +PH | winner |
  |---|---|---|---|---|
  | **fixed-30K** (Object@30K) | **81.75** | 80.12 | 80.38 | **flow** |
  | best-budget (Object@15K) | 81.00 | 81.88 | 79.75 | M3 (selection-dependent) |

- → The DeepONet in-dist "edge" is an artifact of budget selection. Under a fixed budget, **flow wins**. See
  `budget_comparison.csv`.

### 5. Scope & reproducibility caveats now made explicit
- **Robustness (LIBERO-Plus) was measured on LIBERO-Spatial only** (8.3K, 5 seeds). There are **no** robustness
  numbers for Object / Long / Goal — the ">2× more robust" headline is a **single-suite** result.
- **15K checkpoints were deleted** (disk pressure); only their eval JSONs remain. 8.3K and 30K weights are on disk.
- **Per-suite 15K/30K numbers are single-seed** (no error bars). Only Spatial @ 8.3K has 5 seeds + significance.
- **Per-seed CSVs are now committed** next to the headline plots (`runs/report_final/seed_level_spatial_8p3k.csv`),
  not just the PNGs.

---

**Net honest summary.** The DeepONet head's defensible claims are: **~9.6× fewer head params, ~5× faster inference**,
**statistically-tied in-distribution accuracy**, and **significantly higher robustness on LIBERO-Spatial** (the one
suite measured). It does **not** beat flow on a fixed-budget in-dist average, the operator is **not** the source of
in-dist accuracy, and **PH should be dropped** (it hurts robustness).
