# ACT campaign plots — ACT vs ACT+DeepONet vs ACT+DeepONet+PH

Regenerate with `python3 make_plots.py` (from `ACT/`). Latency from `python3 bench_latency.py`.

**Data sources (canonical):**
- In-dist: `act_results/{suite}/runs/eval_rerun_indist/success_rates.json` (10 episodes × 3 seeds)
- OOD / LIBERO-Plus (7 perturbations × 12 tasks): `act_results/{suite}/runs/eval_lerobot_full/robustness_plus.json`
- Latency: `act_results/latency.json` (RTX PRO 6000 Blackwell, bf16, batch=1)
- Params: `run_config.json` + `architecture.md`

| File | What it shows |
|---|---|
| `01_parameters.png` | Total params + backbone/head breakdown (head 37.8M → 10.2M) |
| `02_latency.png` | Planning-forward latency, amortized per-step (replan=5), control Hz |
| `03_indist_overall.png` | In-dist success per suite + average (3-seed error bars) |
| `04_plus_overall.png` | LIBERO-Plus overall robustness per suite + average |
| `05_indist_vs_plus_drop.png` | In-dist vs OOD with robustness drop annotated |
| `06_per_task_indist.png` | Per-task in-dist success, 2×2 grid (all 4 suites × 10 tasks) |
| `07_perturbation_by_category_avg.png` | 7 perturbations, averaged over suites |
| `08_perturbation_per_suite.png` | 7 perturbations, per suite (2×2) |
| `09_perturbation_heatmaps.png` | suite × perturbation heatmap, one per variant |
| `10_perturbation_radar.png` | Radar of 7 perturbations (avg over suites) |
| `11_efficiency_pareto.png` | Accuracy vs latency (bubble=params); accuracy vs params |
| `12_master_summary.png` | Params / latency / in-dist / OOD at a glance |
| `summary.csv`, `summary.json` | All numbers behind the figures |
