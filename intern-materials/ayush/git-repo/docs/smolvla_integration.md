# How the DeepONet head is integrated into SmolVLA

**Source:** `DeepONet PH/v2/modeling_smolvla_deeponet_v2.py` (class `SmolVLADeepONetPolicy`).

## 1. What SmolVLA is

**SmolVLA** (Hugging Face / LeRobot, [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)) is a ~450 M-parameter VLA:

| Part | What | Params |
|---|---|---|
| Backbone | **SmolVLM2-500M-Video-Instruct**, reduced to **16 transformer layers** | ~350 M |
| Action expert | **flow-matching** transformer that denoises an action chunk | ~99.9 M |

Key config: `chunk_size = 50`, `action_dim = 32`, context/prefix hidden dim = `960`, flow inference = ~10 denoise steps.

## 2. The swap

`SmolVLADeepONetPolicy`:

1. **Reuses** the SmolVLM2 backbone and SmolVLA's prefix-token construction (image + state + language → prefix tokens
   `[B, N, 960]`, with a padding mask).
2. **Replaces** the flow-matching expert with the **DeepONet head** through the generic interface:
   ```
   head(prefix_tokens [B, N, 960], pad_mask [B, N])  ->  action_chunk [B, T, 32]
   ```
   The head is **backbone-agnostic**; only `in_proj` (960→512) is SmolVLA-specific. The policy class is the only
   SmolVLA-aware glue.
3. **Loads** the pretrained SmolVLA checkpoint with `from_pretrained(..., strict=False)` so the new head initialises
   fresh while the backbone keeps its pretrained weights.
4. **Freezes the dead flow modules** (`DEAD_PREFIXES = lm_expert`, `action_in_proj/out_proj`, `action_time_mlp`) — they
   are retained in the state dict for checkpoint compatibility but receive no gradient. (Delete them to realise the
   450 M → 360 M storage win.)

## 3. Training recipe (two-stage)

| Stage | Steps | What trains | Why |
|---|---|---|---|
| Stage 1 | 1650 | DeepONet head only (backbone frozen) | warm up the new head before disturbing the backbone |
| Stage 2 | rest (e.g. 13.4K–28.4K) | backbone (low LR) + head (higher LR) | full fine-tune |

- Optimiser: separate LRs (`head_lr ≈ 1e-4`, `backbone_lr ≈ 1e-5`), warmup ~500, **EMA** (0.999) on weights.
- Loss: MSE/L1 regression on the action chunk; `+ λ·PH` for the `M4` variant.
- Crash-safety: `--ckpt_every` writes intermediate checkpoints (after a disk-full incident corrupted a final save).

See `DeepONet PH/v2/train.py` for the full loop and `paper_repro_30k.sh` / the `run_*.sh` scripts for exact invocations.

## 4. Inference / control

- The head produces the full 50-step chunk in **one forward pass** (no denoising loop) → ~29.5 ms vs 148 ms for flow.
- Closed-loop control uses **receding horizon**: predict a chunk, execute the first `--replan` (default 5) actions,
  re-observe, re-predict. The eval harness sets `config.n_action_steps = replan` identically for all models, so flow
  (M1) and DeepONet (M3/M4) are compared under the **same** control loop (`evaluate.py`).

## 5. Porting to other backbones

Because the head is generic, porting = write a new adapter that (a) exposes the backbone's prefix tokens + mask and
(b) sets `in_proj` to the backbone's hidden size. The DeepONet head, PH loss, training loop, and eval harness are
reused unchanged. π0 specifics (PaliGemma 2.9 B + Gemma expert, batch-size/VRAM implications, recipes A/B/C) are in
`DeepONet PH/PORTING_TO_PI0.md`.
