# Authorship & contribution framing (post–Jul 15)

**Date locked:** 2026-07-23  
**Rule:** Anything dated before 2026-07-15 is Ayush’s prior work. Publishable “our” contribution = post–Jul 15 modules + closed-loop gains they cause.

## Split (camera-ready)

| Owner | Owns in the paper |
|---|---|
| **Ayush (pre–Jul 15)** | DeepONet operator action head for SmolVLA (v1→v2), Spatial LIBERO-Plus discovery (38.5% vs 17.9%), efficiency numbers (~9.6× / ~5×), training stack, ACT co-adaptation result |
| **Post–Jul 15 lab contribution (us)** | Chunk-execution (boundary pinning + fractional-τ ensembling), POD eigen-motion trunk, PH negative analysis (drop / anneal), multi-suite Plus completion, Long-gap closing experiments, paper claim rewrite vs NIAF/Spline Policy |

## What we do *not* claim

- Inventing DeepONet-VLA from scratch
- Ayush’s Spatial 5-seed Plus table as our result
- PH as a positive contribution (it hurts robustness; document as negative)
- Frozen pi0.5 transfer as a win (negative result stays his/ours shared finding)

## Minimum bar for co-lead credit (from claim-odds plan)

Need **≥2** of:

1. Closed-loop SR win caused by our execution/trunk (esp. Long ≥ flow ~66.5%, or multi-suite Plus avg >> flow)
2. Named method that is ours (POD trunk and/or pinning/ensemble) with ablation vs plain M3
3. Honest framing as above in the paper

## Campaign artifacts (this run)

- Code: `intern-materials/ayush/contrib/` + `contrib_postjul15/` on blackwell
- Long exec eval: `DeepONet PH/v2/exec_campaign_results/` on blackwell
- Multi-suite Plus: launched on a100 once sync completes
- POD train: after exec signal; Spatial+Long recipes

## Citation / novelty narrowing

Claim **VLM-conditioned branch × (learned or POD) trunk operator head**, not generic “continuous action function” (NIAF 2603.01766, Spline Policy 2606.07386). Always ablate fixed-basis trunk.
