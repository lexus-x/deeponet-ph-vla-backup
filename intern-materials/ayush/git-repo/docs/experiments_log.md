# Experiments log — everything that was run, step by step

This is the chronological record of the project. Outputs referenced are under `DeepONet PH/`.

---

## Phase 0 — Flow-matching baseline (Weeks 1–3)
**Folder:** `Flowmatching PH (Weeks 1-3)/`
- Set up SmolVLA + LeRobot + LIBERO + MuJoCo (EGL) environment.
- Fine-tuned the **flow-matching** SmolVLA baseline on LIBERO; built the closed-loop evaluation harness
  (receding-horizon control, success-rate rollouts).
- Established the metrics: in-distribution success rate, LIBERO-Plus robustness, latency.
- Deliverable incl. `presentation.pptx`.

## Phase 1 — DeepONet v1 (first operator head)
**Folder:** `DeepONet PH/v1_results/`
- First DeepONet action head replacing the flow expert.
- **Result:** matched flow in-distribution but **collapsed on several robustness categories** (camera viewpoints,
  sensor noise → near 0%) while *gaining* on layout/language. The instability motivated the v2 redesign.
- See `DeepONet PH/v1_results/data/robustness_per_category.csv`.

## Phase 2 — DeepONet v2 (main contribution)
**Folder:** `DeepONet PH/v2/` (code), `DeepONet PH/v2_results/` (results)
- Redesigned head: **CrossAttnPool** (8 queries, 3 blocks, d_model 512) + **Branch⊗Trunk** (p=256) + **Fourier-τ**.
- **5-seed head-to-head** on LIBERO-Spatial of:
  - `M1` flow (baseline), `M3` DeepONet, `M4` DeepONet + PH loss.
- Measured in-distribution accuracy, **LIBERO-Plus robustness (7 perturbations)**, inference latency, parameter count.
- **Headline (5 seeds):** flow 79.4±1.4 / 17.9±5.2 robust · M3 80.3±2.4 / **38.5±6.9** robust · M4 81.5±3.9 / 32.6±5.1.
  Head params 99.9 M → **10.4 M**; latency 148 → **29.5 ms**. (`v2_results/data/summary.csv`)

## Phase 3 — Ablations
**Folder:** `DeepONet PH/Ablation_Results/`
- On LIBERO-Spatial, varied one design choice at a time (3–5 seeds each):
  - basis `p`: 256 → 64
  - Fourier-τ: on → off (linear τ only)
  - cross-attn blocks: 3 → 1
  - operator: DeepONet merge → plain **regression head** (same context, no Branch⊗Trunk)
- **Finding:** cross-attn pooling is the most important component; the operator merge is what buys robustness
  (regression head: higher in-dist 83.0 but −6 robustness). (`Ablation_Results/ablations.csv`)

## Phase 4 — Per-suite generalisation (Object, Goal, Spatial @ 15K)
**Folders:** `DeepONet PH/{Spatial,Object,Goal}/`
- Trained & evaluated M1/M3/M4 on each suite (single seed, 15K steps), with **per-task plots** and
  **per-episode videos**.
- In-dist: Spatial 78.5/82.0/80.5 · Object 84.5/**94.0**/87.0 · Goal **93.5**/90.0/89.0 (flow/m3/m4).
- DeepONet's biggest in-dist win is **Object** (+9.5 over flow).

## Phase 5 — task5 investigation (LIBERO-Spatial)
**File:** `DeepONet PH/v2/task5_fix_experiment.py`
- LIBERO-Spatial **task5** ("pick the bowl stacked on the ramekin") is a hard ceiling (~5% flow at 15K) and drags the
  suite average down.
- Confirmed it is **intrinsic difficulty** (elevated stacked-bowl grasp), not data imbalance.
- A **replan sweep** (n_action_steps 5/2/1) did **not** fix it (flat-to-worse) — it is a training/capability ceiling,
  not an inference-tuning issue.

## Phase 6 — Paper-reproduction campaign (30K) — **LIVE**
**Folder:** `DeepONet PH/paper_repro/` · **Script:** `DeepONet PH/v2/paper_repro_30k.sh`
- Goal: push the **flow baseline** to the SmolVLA paper values — **Spatial 90 / Object 96 / Long ≥71** (Goal 92 already
  reached at 93.5%).
- Recipe: **30K steps** (stage1 1650 + stage2 28350), **batch 48**, 20 episodes/task eval, flow-first, per-suite
  train→eval→plot for early incremental results. Intermediate checkpoints (`ckpt_every`) + disk guard after an earlier
  disk-full crash.
- Status streams to `paper_repro/PROGRESS.log`. (Started 2026-06-16; ~30 h flow-only.)

## Phase 7 — Goal object-layout test — **LIVE**
**Folder:** `DeepONet PH/goal_layouttest/` · **Scripts:** `run_goal_seedtest.sh`, `evaluate_seedtest.py`
- **Generalisation / anti-memorisation check** on the frozen Goal flow model (the 93.5% checkpoint).
- LIBERO stores **50 fixed init-state object layouts per task**; the standard eval uses the first 20. This test
  re-evaluates on **3 different layout slices** — `off_0` (init_states[0:20], canonical → reproduces ~93.5%, sanity),
  `off_20` (init_states[20:40]) and `off_30` (init_states[30:50]) — covering all 50 layouts.
- Reports **mean ± σ across object layouts**. Tight σ near 93.5% ⇒ the policy reacts to object positions (generalises),
  not memorises. *(Important correction during this work: varying the env `seed` does NOT move objects in LIBERO — the
  layout is chosen by a fixed init-state table indexed by reset count, so the test varies `init_offset`, not the seed.)*

## Phase 8 — π0 porting analysis
**File:** `DeepONet PH/PORTING_TO_PI0.md`
- Feasibility/time estimate for porting the head to **π0** (PaliGemma 2.9 B + Gemma expert, ~3.3 B, ~7× SmolVLA).
- Recipes A (frozen head-only), B (LoRA), C (full) with VRAM/time implications. The head ports ~unchanged.

---

### Infrastructure notes
- **Disk-full incident:** a full root disk truncated a checkpoint mid-save; fixed by clearing HF caches, adding a
  `guard()` (abort < 10 GB free) and intermediate checkpointing. Keep ≥10–20 GB free.
- **Process management:** long jobs run detached (nohup/systemd, own process group). Kills are by **PID/process-group**,
  never `pkill -f <scriptname>` (which self-matches the kill command and orphans the trainer).
