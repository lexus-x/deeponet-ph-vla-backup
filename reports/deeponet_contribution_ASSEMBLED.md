# DeepONet — assembled lab contribution (post–Jul 15)

**Updated:** 2026-07-24 (afternoon). Supersedes the "we need an SR win" framing.
Companion docs: [authorship framing](deeponet_authorship_framing.md) ·
[contribution restart](deeponet_contribution_restart.md) ·
[campaign results (FINAL)](deeponet_campaign_results.md).

## Thesis (honest, defensible, no baseline-beating required)

Ayush owns the DeepONet operator action head + Spatial LIBERO-Plus discovery +
efficiency (pre–Jul 15). **Our** contribution clears the ≥2-of-3 co-lead bar
*without* an SR win:

- **(2) Named methods, ours** — POD eigen-motion trunk + Operator-BID
  (DeepONet-native bidirectional decoding), each with a matched-budget ablation
  vs plain Fourier-M3.
- **(3) Honest framing** — per authorship split; PH stays a negative result; no
  rebranding of pre–Jul-15 work.
- **Completion** — multi-suite LIBERO-Plus (Object/Long/Goal), the repo's own #1
  flagged hole. **Required a harness bugfix** (below) before it was computable.

A win on bar **(1)** is a long shot; the only live shot is multi-suite Plus
m3 ≫ flow on ≥2 suites (Spatial already +0.125). Do not hang credit on it.

## Contribution #0 — multi-suite Plus was broken; we root-caused + fixed it

The overnight multi-suite run reported Object=0.0 and Long=0.0 for **both** m3
and flow — an artifact, not a result. Root cause: `evaluate_plus.py` defaulted
`--dataset` to `lerobot/libero_spatial_image` (`REPO_DATA`) regardless of
`--suite`, so Object/Long-trained policies were de-normalized with **Spatial's**
action stats (means/stds differ substantially across suites) → garbage actions →
every rollout fails. Spatial worked only because its stats happened to match.

**Fix (single point, shared harness):** `evaluate_plus.py` now derives the norm
dataset from `--suite` (`SUITE_DATASET` map) unless `--dataset` is given, and
prints `[plus] norm dataset = … (suite=…)`. Backup: `evaluate_plus.py.bak_*`.
**Smoke-verified:** Object m3 on the exact indices that logged `x` at 0.0 now
returns OK (4/4 on Light + Language). This bugfix is itself honest completion
work — without it the multi-suite table is uncomputable.

## Results table

In-dist (10 eps) and efficiency are Ayush's; robustness/POD/BID are ours.

| Axis | M3 (ours-improved / DeepONet) | Flow | Read |
|------|------|------|------|
| Spatial in-dist SR | 77 | 78 | tie (his) |
| Long in-dist SR (10ep) | 54 | 66 | flow +12 (his) |
| **Plus Spatial** avg | **0.446** | 0.321 | **m3 +0.125** ✓ |
| **Plus Object** avg | _PENDING (PID 197211)_ | _PENDING_ | fill |
| **Plus Long** avg | _PENDING (PID 197212)_ | _PENDING_ | fill |
| POD-30K Spatial SR (matched budget) | _PENDING (POD eval running)_ | 78 | vs Fourier-M3 77 |
| Operator-BID Long SR | _PENDING (a100: bid_long_a100)_ | 66 | vs m3_r5 60 |

Plus Spatial per-category (m3): Camera .50 · Light .625 · Sensor .25 ·
BgTex .375 · ObjLayout .625 · RobotInit .125 · Language .625.

### Prior (failed) win-bets — kept as negative science
- Long ensemble 55.5 (< m3_r5 60), Long pin ~5 (catastrophic): temporal
  averaging/pinning across mode switches hurts (BID-consistent).
- POD-8.3K Spatial 58 (−19 vs 77): **confounded** (8.3K vs 30K), not a fair
  ablation — POD-30K rerun fixes this.

## What each pending number buys

- **POD-30K ≈ 77** → fair trunk ablation lands (bar 2), "data-driven trunk ≈
  Fourier here" is a clean, citable finding regardless of sign.
- **BID > 60** → a genuine closed-loop win (bar 1) + named method (bar 2). Even
  a tie keeps the *method* as a contribution (first multi-hypothesis decoder for
  a deterministic operator head; diffusion-BID doesn't apply).
- **Plus Object/Long m3 ≫ flow** → robustness generalizes → headline. **If not**
  (Long favors flow in-dist, so plausible), the honest table shows the edge is
  Spatial-specific — still publishable completion, but narrows the claim.

## Novelty narrowing (keep)

Claim **VLM-conditioned branch × (Fourier or POD) trunk operator head** for
SmolVLA + efficiency, not generic "continuous action function." Always cite +
ablate vs NIAF (2603.01766) and Spline Policy (2606.07386); fixed-basis trunk
ablation is the POD/Fourier comparison above.

## To finalize (fill 4 cells)
1. POD-30K Spatial → `pod30k_eval_spatial/success_rates.json`
2. Operator-BID Long → **a100** `~/deeponet_campaign/bid_long_a100/m3_bid_t{0-4,5-9}/`
3. Plus Object m3+flow → `plus_multisuite_fixed/libero_object/`
4. Plus Long m3+flow → `plus_multisuite_fixed/libero_10/`
