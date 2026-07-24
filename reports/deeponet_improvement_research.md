# Improving Ayush's DeepONet Action Head — Research Report (2026-07-17)

Supervisor research memo for the `DeepONet-PH-VLA` track (Ayush). Inputs: the repo's own results
(`intern-monitor/.cache/ayush_deeponet_new.md`, current as of the 30K campaign, 2026-06-21) + a
12-agent literature workflow (6 research threads, each adversarially citation-verified — every
arXiv ID below was checked against its abs page this session; corrections noted inline).

**Scope note:** DeepONet/PH is Ayush's toolbox. These are supervision suggestions for *his* model —
none of this feeds the supervisor's own ACE-DRO track.

---

## 1. What the model is (10-second recap)

SmolVLA with its 99.9M flow-matching action expert swapped for a 10.4M DeepONet head:
`CrossAttnPool(prefix tokens) → Branch → c ∈ ℝ²⁵⁶`, `Trunk(τ + Fourier feats) → φ(τ) ∈ ℝ²⁵⁶`,
`a(τ) = OutMLP(c ⊙ φ(τ))`. One forward pass per chunk (29.5ms vs 148ms). Headline: matches flow
in-distribution, **>2× more robust** on LIBERO-Plus (38.5 vs 17.9), 9.6× smaller head.

## 2. Where it is weak — from Ayush's own numbers

| # | Weakness | Evidence |
| --- | --- | --- |
| W1 | **LIBERO-Long: loses to flow by 6–8 pts** (58.5/60.5 vs 66.5) | 30K campaign; per-task scores bimodal; highest train-MSE (0.041). |
| W2 | **Deterministic, unimodal head** | Candidate cause for W1 — but see §6: the literature now argues *against* this being the root cause. |
| W3 | **PH loss (M4) is not Pareto-optimal** | −5.9 robustness (32.6 vs 38.5), Camera Viewpoints collapse (26.7→14.7), trails M3 on 4-suite avg. §7 has a literature-backed mechanism. |
| W4 | **Sensor Noise is the only LIBERO-Plus category lost** (21.3 vs 25.3) | v1 also collapsed on camera/sensor. |
| W5 | **"Full" config isn't the best config** | Own ablation: removing Fourier-τ improved *both* metrics (82.7/39.0 vs 80.3/38.5; 3 seeds, ±8.7). |
| W6 | **Flow baseline repro far below published** (Spatial 79.5 vs 90; avg 81.0 vs 87.3) | §4 now explains this concretely: ~4.4× fewer training samples than the official recipe + known dataset/eval pitfalls. |
| W7 | **Single-seed everywhere except Spatial**; Object overfits at 30K (94.0→87.0); post-hoc "best-budget checkpoint" selection | Reviewer bait. |

## 3. Four headline findings from the verified literature sweep

1. **The novelty window is closing.** *NIAF — Neural Implicit Action Fields* (ICML 2026,
   arXiv 2603.01766) already gives VLAs one-pass, continuous-time action functions (VLM-modulated
   SIREN weights) with reportedly ~97.9% LIBERO avg incl. ~94% on Long (numbers from beyond the
   abstract — check the PDF before treating as the bar). *Spline Policy* (2606.07386) already
   decodes chunks as fixed-basis × predicted-coefficients across VLA backbones. **No true
   collision exists** — nothing uses an operator-learning branch⊗trunk head in a vision-based
   IL/VLA policy (nearest: branch-trunk BC for control laws without vision/language, 2604.03449) —
   but the claim must narrow from "continuous-function action head" to **"operator-learning
   (VLM-conditioned branch × learned trunk) action expert"**, and the paper must cite + ablate
   against a fixed-basis (spline/DCT) trunk to prove the learned trunk earns its place.
2. **W6 is explained: the baseline (and the DeepONet runs) are undertrained on a possibly-broken
   setup.** The SmolVLA paper's verified LIBERO recipe is 100K steps × batch 64 (~6.4M samples;
   the intern used 30K × 48 ≈ 1.44M), 512×512 images, chunk 50, frozen VLM, cosine 1e-4→2.5e-6,
   replanning after *every* executed action. Community repro threads add three silent killers:
   corrupted LIBERO v2.1→v3.0 dataset conversion (fix: re-download `HuggingFaceVLA/libero`),
   MuJoCo must be 3.3.2 (older versions render different colors), and a train/eval
   `n_action_steps` mismatch that **disproportionately hurts LIBERO-Long** — so part of W1 may be
   protocol artifact, not architecture. Sources: arXiv 2506.01844 (recipe verified in HTML full
   text), `lerobot` issues #1369 and #3264, LeRobot LIBERO docs.
3. **W3 has a clean mechanism in the literature.** Vietoris–Rips persistence on a ~29-point cloud
   in ℝ³² is exactly the high-dimension-low-sample-size regime where persistence diagrams are
   provably noise-dominated (*Curse of Dimensionality on Persistence Diagrams*, 2404.18194;
   spectral-methods fix, NeurIPS 2024, 2311.03087), and PH-loss gradients touch only a few
   critical simplices (2203.16748, 2405.18820). Plausible (not yet verified on this model): the
   loss anchors the policy to a brittle, training-view-specific topological signature — which is
   consistent with the camera-viewpoint collapse. Fixes in §7.
4. **Don't assume W1 is multimodality.** OpenVLA-OFT (2502.19645) reaches ~95% on LIBERO-Long
   with a *plain L1 regression head* + parallel decoding — deterministic heads per se don't fail
   on Long. Run diagnostics (§6) before building a multimodal head.

## 4. Fix the recipe and eval protocol first (attacks W6, W7, and part of W1)

Everything else in this memo is noise until this is done, because it moves both models and every
robustness number:

1. **Environment audit:** re-download the `HuggingFaceVLA/libero` dataset (v3.0 conversion bug),
   pin MuJoCo 3.3.2 (`lerobot` #1369).
2. **Eval harness audit:** `eval.batch_size=1` (initial-state-diversity bug #2850), eval
   `n_action_steps` matched to training chunk config (#3264 — mismatch hits Long hardest,
   a user saw 56% on libero_10 from this alone), fixed episode count (10/task is ±15–20% noisy).
3. **Budget-matched retrain:** 100K steps × batch 64, 512×512, chunk 50 for flow AND m3/m4.
   If GPU-time forbids it, keep 30K but state "compute-matched, ~4.4× below the published
   budget" in every table.
4. **Replan protocol:** the SmolVLA paper replans after every executed action; the intern evals
   at replan 5. ~~At 29.5ms the DeepONet head can afford replan-1 trivially — report both, it
   turns the latency win into an accuracy claim.~~ **Refuted 2026-07-21 (see §6): replan-1
   measures 44.5 % vs 60.0 % at replan-5 on Long.** The latency win does NOT convert into an
   accuracy win by replanning alone. Keep replan-5 as the headline protocol; if comparing
   against the paper's per-step-replan protocol, report the gap and note that chunked
   execution is load-bearing for this head.

## 5. Head-architecture upgrades from the DeepONet literature (all IDs verified)

Ranked by (expected gain ÷ implementation cost) for this 10.4M head:

| Rank | Technique | Paper | What to do here |
| --- | --- | --- | --- |
| 1 | **POD/PCA trunk** | POD-DeepONet, in the FNO-vs-DeepONet fair comparison, CMAME 2022 (2111.05512) | PCA basis of demo action chunks as fixed (or init) trunk; branch predicts coefficients. Cheapest run; gives interpretable "eigen-motions" figure. |
| 2 | **Two-step training** | Lee & Shin, SIAM J. Sci. Comput. 2024 (2309.01020) | Trunk-first fit + Gram–Schmidt/QR orthonormalization, then branch against the frozen basis. Repo already has a two-stage schedule — this is a reorder. |
| 3 | **Diagnose branch vs trunk before spending params** | Heinlein & Taraz error decomposition, 2026 (2602.21910) | Their decomposition says branch error dominates (when internal dim is large) — consistent with Ayush's own ablation (pooler blocks mattered most; trunk tweaks marginal). Run it, then invest in the pooler/branch. |
| 4 | **Modified DeepONet** | Wang, Wang & Perdikaris, J. Sci. Comput. 2022 (2110.01654) | Gated encoder fusion into branch/trunk layers + NTK-style loss reweighting (could target underweighted late-chunk timesteps). ~30 lines. |
| 5 | **Shift-DeepONet-style τ warp** | Lanthaler et al., ICLR 2023 (2210.01074) | Branch-conditioned scale/shift of τ before the trunk — input-adaptive time-warping for gripper open/close discontinuities and re-timed long segments. |
| 6 | **S-DeepONet recurrent branch** | He et al., Eng. Appl. AI 2024 (2306.08218) | GRU branch over recent pooled VLM states so c carries history — path-dependence fix aimed at Long. (Frame as recurrent conditioning, NOT "memory module" — associative memory is Aryan's lane.) |
| 7 | **flexDeepONet canonical-frame pre-transform** | Venturi & Casey (2204.12670) | Small pre-net canonicalizes conditioning before branch⊗trunk — a targeted *hypothesis* for camera-viewpoint robustness (PDE evidence only). |
| 8 | **NOMAD nonlinear decoder** | Seidman et al., NeurIPS 2022 (2206.03551) | `MLP([c; φ(τ)])` instead of product+MLP — one-line ablation; the current OutMLP is already halfway there. |
| 9 | **HyperDeepONet** | Lee et al., ICLR 2023 (2312.15949) | Branch emits (low-rank) trunk weights; most expressive per param but biggest rewrite. |
| 10 | **RaNN least-squares readout** | 2025 (2503.00317) | Closed-form ridge refit of the final layer post-training — near-free calibration diagnostic. |

Also verified but lower priority: MIONet multi-branch factorization (2202.06137) — separate
vision/language/proprio branches merged by product, so a perturbed modality degrades one factor
(relevant to Sensor Noise and Language robustness); DeepOKAN RBF-KAN trunk (2405.19143).

## 6. LIBERO-Long: diagnose, then fix (W1, W2)

> **2026-07-21 — MEASURED ON GPU. The P0 "replan-1" recommendation below is REFUTED.**
> Ran on the lab Blackwell using Ayush's own trained m3 30K Long checkpoint, 10 tasks ×
> 20 episodes, his `evaluate.py --replan`:
>
> | config | replan | LIBERO-Long mean SR |
> | --- | --- | --- |
> | m3_r5 (his default) | 5 | **60.00 %** ← reproduces his reported 58.5 %, harness validated |
> | m3_r1 (my proposal) | 1 | **44.50 %** |
>
> Replanning every step is **15.5 points WORSE**, and replan-5 wins **8 of 10 tasks**
> (ties 1, loses 1). So "the operator head is fast, so replan every step and close the
> Long gap" is wrong — it moves m3 *further* from flow's 66.5, not closer. This is
> consistent with the chunking literature (BID 2408.17355: chunking trades reactivity
> for temporal consistency) and with Exp A in
> `intern-materials/ayush/contrib/RESULTS.md`, where naive per-switch behaviour did not
> average out systematic per-replan error.
>
> **What this redirects the work toward:** the failure mode at replan-1 is per-step mode
> thrash with no smoothing, which is exactly what temporal ensembling and boundary
> pinning exist to fix. The live question is now "is per-step *observation freshness*
> bad, or is *unsmoothed switching* bad?" — test replan-1 **+ fractional-τ ensembling /
> boundary pinning** (`operator_chunk_exec.py`) against these two anchors. Plain replan-1
> is settled: it loses.

**Diagnose first — three cheap tests on existing checkpoints:**

- **Protocol check** (§4.2): eval `n_action_steps` mismatch is known to hit Long hardest.
- **Boundary-artifact metric** (2603.11642): log `|a_new(0) − a_old(τ_switch)|` per chunk
  boundary on Long successes vs failures — tests whether the gap is boundary discontinuity.
- **Multimodality check** (Mazza et al. 2026, 2605.22493): measure how multimodal demo chunks
  actually are at states where m3 diverges from flow. OpenVLA-OFT's ~95% Long with plain L1
  (2502.19645) says regression heads don't inherently fail here — so don't build a multimodal
  head on faith.

**Fix menu, ordered by cost (several become *easier* for a continuous-τ head — engineering
hypotheses, not published results):**

1. **Replan-1 closed-loop eval** — free at 29.5ms; kills within-chunk open-loop drift, which
   BC theory (2507.09061) identifies as the compounding-error mechanism chunking mitigates.
2. **ACT-style temporal ensembling** (2304.13705) — average overlapping chunks; with a cached
   code c, extra τ queries are just Trunk+OutMLP evaluations, and fractional-τ alignment avoids
   the time-quantization error of ensembling discrete chunks.
3. **PACE-style phase-aware replan boundaries** (2606.00537) — replan at low-speed points;
   `da/dτ` is available analytically by autodiff through the trunk.
4. **Training-time chunk-boundary conditioning** (Black et al. 2025, 2512.05964 — the zero-runtime
   successor to RTC 2506.07339): condition the branch on the committed action prefix / previous
   code c_prev + phase offset, with simulated switch points. Best training-time bet; can even
   hard-constrain `a_new(0) = a_old(τ_switch)`.
5. **A2C2-style residual correction head** (2509.23224): tiny `δ(o_t, τ)` net reusing trunk
   features, runs every step on top of the frozen operator — attacks Long AND sensor noise
   (+23/+7 over RTC in their abstract, verified).
6. **Dynamic execution horizon** (2606.11408): tiny head on c predicting when to re-query.

Not transferable (need iterative/stochastic samplers): RTC's inpainting, BID's resampling
(BID's current arXiv title: *Bidirectional Decoding: Improving Action Chunking via Guided
Test-Time Sampling*, 2408.17355 — the "Closed-Loop Resampling" subtitle is stale).

**If multimodality IS confirmed** (§6 diagnostic): smallest-diff option is **K branch codes +
winner-take-all training** (Rupprecht et al., ICCV 2017, 1612.00197) with a softmax gate that
recent theory shows yields calibrated mode probabilities (ICML 2024, 2406.04706). Still one
forward pass. Heavier: VQ-BeT-style residual quantization of c (2403.03181). Orthogonal ablation
worth one run: MeanFlow-style one-step displacement regression (2603.01469 — a one-step SmolVLA
variant, also a must-cite baseline). Avoid EBM/IBC (2109.00137 — unstable contrastive training,
per Diffusion Policy's analysis 2303.04137) and expect mode-blurring from one-step consistency
distillation (2405.07503, 2410.21257).

## 7. Fixing the PH story (W3)

> **2026-07-17 addendum — tested, partially refuted.** The H0-swap recommendation below
> was implemented and measured on 3,379 real LIBERO-Spatial chunks
> (`intern-materials/ayush/contrib/RESULTS.md`, Exp B): TRUE H0 is *worse* than Ayush's
> top-k surrogate as a shape discriminator (AUC 0.63 vs 0.92) because MST deaths are
> fine-scale and noise-sensitive, while his largest-distance spectrum is stable. Keep
> the surrogate (optionally + PCA-3, AUC 0.929); the M4 viewpoint collapse is better
> explained by shape-anchoring than by noise — fix with λ_PH annealing, not a different
> filtration.

Literature-backed priority order (mechanism in §3.3 — plausible, should be verified on-model):

1. **Drop VR-H1 on ℝ³² → H0 sublevel filtration on per-dim action signals** (topology layer,
   1905.12200). H0 is the stable feature, and the successful robotics PH precedent is H0
   components (Vieira et al., ICRA 2022, 2202.02937). For temporal structure, sliding-window
   (Takens) embeddings are the principled route (Perea & Harer, 1307.6188).
2. **Reduce dimension before filtration**: normalized PCA to 2–3 dims (2404.18194) or
   spectral/diffusion distances on the chunk kNN graph (2311.03087).
3. **Gradient/convergence hygiene**: sliced-Wasserstein diagram distance (ICML 2017, 1706.03358),
   densified subgradients (2203.16748; diffeomorphic interpolation, NeurIPS 2024, 2405.18820),
   total-persistence regularizer (+ λ annealing) for stable convergence (2206.02946 — note:
   arXiv preprint, NOT ICML as sometimes cited). Foundations: Carrière et al., ICML 2021
   (2010.08356) — check `ph_loss.py` meets its convergence condition.
4. **Verify the mechanism**: compare M3 vs M4 trajectories on the same camera-shifted rollouts;
   if M4's trajectories snap to canonical training shapes, the diagnosis holds and makes a good
   paper subsection (turns W3 from a liability into an analyzed finding).

## 8. Revised priority plan

| Pri | Action | Cost | Attacks |
| --- | --- | --- | --- |
| P0 | Environment + eval audit: re-download LIBERO v3.0, MuJoCo 3.3.2, `eval.batch_size=1`, matched `n_action_steps` | hours | W6, part of W1 |
| P0 | Replan-1 + temporal-ensembling eval on Long (existing ckpts) | hours, eval-only | W1 |
| P0 | Boundary-artifact + multimodality diagnostics on Long | hours | W1 vs W2 attribution |
| P0 | Read NIAF (2603.01766) + Spline Policy (2606.07386); narrow the novelty claim; plan fixed-basis ablation | reading | positioning |
| P1 | Budget-matched retrain at 100K×64 recipe (flow + m3 + m4) | GPU-days | W6, likely lifts everything |
| P1 | POD trunk; two-step training; no-Fourier-τ default re-check (5 seeds) | 1–2 runs each | W5, stability |
| P1 | PH: H0-sublevel + PCA-subspace variant + viewpoint-collapse diagnosis | 1–2 runs | W3 |
| P1 | Sensor-noise augmentation at train time | 1 run | W4 |
| P2 | Chunk-boundary conditioning (2512.05964-style); A2C2 residual head | 2–3 runs | W1 |
| P2 | Modified-DeepONet fusion; shift-τ warp; MIONet split branches | 1–2 runs each | accuracy, W4, Language |
| P2 | K-code WTA head — ONLY if the multimodality diagnostic comes back positive | 2–3 runs | W2 |
| P3 | 5-seed everything; pre-registered checkpoint-selection rule | GPU budget | W7 |

## 9. Prior-art & citation status

- **Novelty verdict:** no direct collision found (adversarial search, 12 near-misses mapped:
  NIAF 2603.01766, Spline Policy 2606.07386, operator-BC-for-control 2604.03449, CINOC
  2605.25867, BEAST 2506.06072, FAST 2501.09747, FreqPolicy 2506.01583, FCNet 2405.19885,
  NDP 2012.02788, Movement Primitive Diffusion 2312.10008, KOROL 2407.00548, DeepONet-MPC
  2505.18008). Negative claims are non-exhaustive; re-run before submission.
- **All ~50 citations in this memo were verified against arXiv abs pages / GitHub / docs this
  session.** Corrections found: BID's title changed on arXiv; 2206.02946 is a preprint (not
  ICML 2022); "SW1PerS" is a companion paper (1505.02033), not the title of 1307.6188.
- Caveats carried from the verifiers: NIAF's exact LIBERO numbers are body-text claims not
  checked beyond the abstract; all "this transfers to our head" statements are design
  hypotheses, not demonstrated results; PDE-benchmark gains (modified DeepONet, flexDeepONet)
  may not transfer to action decoding.
