# Porting DeepONet + PH from SmolVLA to π0 (and π0.5)

## TL;DR
lerobot ships `pi0`, `pi05`, and `pi0_fast`. lerobot's π0 is structurally a
**scaled-up twin of SmolVLA** — same flow-matching + action-chunk design, same
`embed_prefix`/`embed_suffix`, same `action_in_proj`/`action_out_proj`/`state_proj`,
same `chunk_size=50`, `max_action_dim=32`. So our DeepONet head and PH loss port
**almost unchanged**; only the backbone/expert names and the context source differ.
The real cost is **compute**: π0 is ~3.3B vs SmolVLA ~0.45B (~7×).

## 1. Structural parallel (what makes it a near-drop-in)

| Piece | SmolVLA (`modeling_smolvla.py`) | π0 (`modeling_pi0.py`) |
|---|---|---|
| Flow module | `VLAFlowMatching` | `PI0FlowMatching` (class ~553) |
| Backbone (VLM) | `vlm_with_expert.vlm` (SmolVLM2, ~350M) | `paligemma_with_expert.paligemma` (PaliGemma, ~2.9B) |
| Action expert | `vlm_with_expert.lm_expert` (~98M) | `paligemma_with_expert.gemma_expert` (Gemma expert) |
| Prefix embed | `embed_prefix(...)` | `embed_prefix(...)` (line 644) |
| Suffix embed | `embed_suffix(x_t, t)` | `embed_suffix(state, x_t, t)` (line 687) |
| Action proj | `action_in_proj`/`action_out_proj`/`state_proj` | same names (lines 579-582) |
| Loss / sampling | `forward` + `sample_actions` (10-step denoise) | `forward` + `sample_actions` (10-step denoise) |
| chunk_size / dims | 50 / state32 / action32 | 50 / state32 / action32 (identical) |

Because the dims match, **the DeepONet head config is literally unchanged**
(`context_dim`=text hidden size, `chunk_size=50`, `action_dim=32`).

## 2. Exact code changes (estimated ~1 day of work)

Copy `modeling_smolvla_deeponet.py` -> `modeling_pi0_deeponet.py` and change:

1. **Base classes**: `SmolVLAPolicy/VLAFlowMatching` -> `PI0Policy/PI0FlowMatching`.
2. **Backbone prefix**:
   `"model.vlm_with_expert.vlm."` -> `"model.paligemma_with_expert.paligemma."`
3. **Dead prefixes** (the now-unused flow expert + projections):
   `"...lm_expert."` -> `"...gemma_expert."`; keep the `action_*_proj` entries.
4. **Context extraction** — π0 is *cleaner*: it already exposes a prefix-only path
   via `paligemma.model.language_model.forward(...)` (line ~466). Run prefix-only,
   masked-mean-pool the returned hidden states -> context vector. (On SmolVLA we
   had to force the self-attention path with `fill_kv_cache=True`; on π0 the
   language_model forward is directly callable.)
5. **context_dim** = `paligemma.config.text_config.hidden_size` (PaliGemma text width).

Unchanged: `deeponet_head.py`, `ph_loss.py`, the two-stage trainer, EMA, the
`evaluate.py`/`evaluate_plus.py` rollout + LIBERO-Plus harness (π0 uses the same
lerobot pre/post processors and `select_action` queue, so replan-5 works as-is).

## 3. Compute — the real cost

π0 backbone (PaliGemma ~2.9B) dominates. VRAM for **full** fine-tune at bf16:
weights ~6GB + grads ~6GB + AdamW states ~24-36GB + EMA ~12GB + activations ->
**~70-90GB**, so batch 48 likely will NOT fit in 98GB; expect **batch 8-16 +
grad accumulation**. Three realistic recipes:

| Recipe | What trains | VRAM | Time / run (1 model, 1 seed) | Notes |
|---|---|---|---|---|
| **A. Frozen backbone, head-only** | DeepONet head (2-10M) only | ~25-35GB | ~1-2h | Cheapest; tests if a frozen PaliGemma + DeepONet operator is enough. Likely lower accuracy. |
| **B. LoRA backbone + DeepONet head** | LoRA adapters (~30-100M) + head | ~40-55GB | ~4-8h | Best cost/quality trade-off for π0. Recommended. |
| **C. Full fine-tune** | full 2.9B backbone + head | ~70-90GB (batch 8 + grad-accum) | ~10-18h | Strongest, but multi-seed = many days on one GPU. |

For a **defensible multi-seed study** on π0, recipe **B (LoRA)** with 2 seeds x
3 models (M1/M3/M4) ≈ 6 runs x ~6h ≈ **~36h training** + ~12h eval. That already
exceeds the 36h cap, so π0 realistically means: **fewer seeds (1-2), LoRA not
full-FT, and/or one benchmark** — or a bigger/multi-GPU machine.

## 4. Why π0 is still worth it
- The DeepONet **latency win grows**: π0's flow head runs 10 denoise steps through
  a larger expert; replacing it with a single DeepONet pass should beat the 6.35×
  we measured on SmolVLA.
- Higher accuracy/robustness ceiling -> cleaner action chunks -> the PH topology
  term is more meaningful.
- Same code, so results are directly comparable across SmolVLA and π0 (a nice
  "does the idea scale?" axis for the thesis).

## 5. Risks / caveats
- VRAM: full-FT batch 48 won't fit; must drop batch + grad-accum or use LoRA.
- The DeepONet "dead" gemma_expert is bigger -> if kept-but-frozen it wastes more
  VRAM; for π0 it's worth actually deleting it (more surgery, but real savings).
- π0.5 adds hierarchical/knowledge-insulation structure -> more integration work
  than π0; do π0 first.
- None of this is benchmark-proven for DeepONet+PH yet; SmolVLA results (in
  progress) should justify the π0 spend before committing the GPU-weeks.

## 6. Recommended sequence
1. Finish the SmolVLA study (in progress) -> decide if DeepONet+PH holds up.
2. If yes: port to **π0 with LoRA (recipe B)**, 1-2 seeds, LIBERO-Spatial +
   LIBERO-Plus, same eval harness.
3. Only escalate to full-FT / π0.5 if a stronger headline is needed and more
   compute is available.
