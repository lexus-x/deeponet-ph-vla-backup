# DeepONet campaign results — FINAL (2026-07-24)

## Bottom line

All three post–Jul 15 “win” bets **missed**. Stock Spatial remains a tie; Long still favors flow; execution fixes and POD trunk both hurt vs Fourier M3.

| Bet | Result | vs anchor |
|-----|--------|-----------|
| Long ensemble | **55.5%** | −4.5 vs m3_r5 60 |
| Long pin | ~**5%** (partial) | catastrophic |
| POD Spatial (8300, 10eps) | **58.0%** | **−19** vs M3 77 / Flow 78 |
| Spatial M3 vs Flow | 77.0 vs 78.0 | tie |
| Long M3 vs Flow (10eps) | 54.0 vs 66.0 | flow +12 |

## POD detail (blackwell)

Per-task %: 100, 50, 70, 50, 10, 20, 100, 80, 70, 30 → mean **58%**  
Train: 8300 steps, 140.6 min, ckpt `pod_train_spatial/checkpoints/8300`

## Claim odds (honest)

| Path | Odds now |
|------|----------|
| Co-lead via measurable post–Jul 15 **wins** | **down sharply** — no SR win to show |
| Diagnostics / negative results / honest ablation | still publishable as contribution, not lead invention |
| Floor (completion/diagnostics only) | ~**15–25%** lead-level credit |
| Sole inventor of DeepONet-VLA | **no** (unchanged) |

Ayush keeps operator head + Spatial discovery. Your post–Jul 15 work is **execution diagnostics + POD attempt + multi-suite confirms** — valuable as completion/negative science, not as a beating baseline.

## Authorship

See `reports/deeponet_authorship_framing.md`. Do not rebrand pre–Jul 15 as sole invention.

## Artifacts

- blackwell ens/pin: `.../v2/exec_campaign_results/SUMMARY.json`
- a100 Spatial: `~/deeponet_campaign/plus_spatial_results/SUMMARY_spatial_indist.json`
- a100 Long: `~/deeponet_campaign/long_confirm_results/SUMMARY_long_confirm.json`
- POD: `.../v2/pod_eval_spatial_8300/SUMMARY.json`
