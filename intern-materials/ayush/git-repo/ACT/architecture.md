# ACT + DeepONet — Architecture

Detailed architecture of the ACT-backbone policy used in this study, in three variants:
**ACT** (baseline) · **ACT + DeepONet** (operator action head) · **ACT + DeepONet + PH** (operator head + gated
persistent-homology loss). Implementation: [`modeling_act_deeponet.py`](modeling_act_deeponet.py),
[`deeponet_head_v2.py`](deeponet_head_v2.py), [`lang_encoder.py`](lang_encoder.py),
[`ph_loss_gated.py`](ph_loss_gated.py).

---

## 1. Observation → action, at a glance

```
  obs ──► [ResNet-18]  [LangEncoder]  [state proj]  (+ CVAE z, train-only)
              │             │              │
              └──────► Transformer ENCODER (4 layers, d=512) ──► memory [B, N, 512]
                                                                     │
                          ┌──────────────────────────────────────────┤
                ACT baseline                              ACT + DeepONet (+PH)
          Transformer DECODER (7 layers)            DeepONet operator head
                          │                                  │
                          ▼                                  ▼
                action chunk  â  ∈ ℝ^[T=100 × 7]   (executed receding-horizon, replan k)
```

The policy maps the current observation to a **whole action chunk** `â ∈ ℝ^{T×7}` (T = 100 timesteps, 7-DoF
delta end-effector + gripper). At test time only the first `k` actions of each chunk are executed before
re-planning (receding horizon; `k` = `n_action_steps`, default 5), using EMA weights.

---

## 2. Inputs (LIBERO observation, fps 10)

| Input | Shape | Encoder |
|---|---|---|
| Agentview RGB | `3 × 256 × 256` | ResNet-18 (shared) |
| Wrist RGB | `3 × 256 × 256` | ResNet-18 (shared) |
| Proprio state | `8` = eef_pos(3) + eef_axis-angle(3) + gripper_qpos(2) | linear projection → 512 |
| Language instruction | text | `TinyLanguageEncoder` → 1 token (512) |

Images are 180°-flipped to match the training render convention; state is normalised with dataset stats.

---

## 3. Encoder stack

### 3.1 Visual backbone — ResNet-18 (≈ 11.2 M, shared across both cameras)
ImageNet-initialised ResNet-18 (final avg-pool/fc removed) produces a spatial feature map per camera, flattened
to a sequence of visual tokens and projected to `d_model = 512` (+ 2-D sinusoidal position embedding).

### 3.2 Language — `TinyLanguageEncoder` (≈ 4 M)
- **BERT tokenizer only** (no pretrained BERT weights) → token ids.
- Fresh **128-d embedding** table, then **2 transformer encoder layers**.
- Mean/CLS-pooled to a single **language token** projected to `d_model = 512`, prepended to the encoder input.
- *(Note: this is a small from-scratch language encoder, not "BERT-tiny".)*

### 3.3 Proprioceptive state
Linear projection of the 8-d state vector to a single state token (512).

### 3.4 CVAE style latent `z` (training only)
ACT is a **CVAE**. During training a small encoder reads the ground-truth action sequence (+ state) and outputs
a **32-d style latent** `z` (reparameterised), prepended as a token to inject multimodality; a **KL term**
regularises `z` toward `N(0, I)`. **At inference `z = 0`** (the prior mean), so the CVAE encoder is dropped.

### 3.5 Transformer encoder
- **4 layers**, `d_model = 512`, `dim_feedforward = 3200`, multi-head self-attention, pre-norm, GELU.
- Input tokens = [`z`] + [language] + [state] + [visual tokens (both cameras)].
- Output: a memory sequence `memory ∈ ℝ^{B × N × 512}` consumed by the action head.

---

## 4. Action heads

### 4.1 ACT baseline — Transformer decoder (≈ 37.8 M)
A **7-layer transformer decoder** with `T = 100` learned positional queries cross-attends to `memory` and emits
the action chunk `â ∈ ℝ^{T × 7}`. This is the head the DeepONet variant **replaces**.

### 4.2 DeepONet operator head (≈ 10.2 M) — replaces the decoder
A DeepONet approximates an **operator** `G : u ↦ G(u)(τ)` as **Branch(u) ⊗ Trunk(τ)**. Here `u` = the encoded
observation, and `τ ∈ [0,1]` indexes the chunk timestep, so the head outputs the whole chunk in one pass.

```
 memory [B,N,512]
      │
      ▼
 ┌─────────────────────────────────────────────┐
 │ CrossAttnPool                                │   8 learned queries,
 │   Perceiver-style cross-attention × 3 blocks │   → pooled context  u ∈ ℝ^{B × 8 × 512}
 └───────────────────┬─────────────────────────┘
                     │
        ┌────────────┴───────────────┐
        ▼                            ▼
   BRANCH b(u)                   TRUNK φ(τ)
   MLP → p = 256 coeffs          τ ∈ [0,1]^T → Fourier features (n = 6)
   c ∈ ℝ^{B × 256}                → MLP → φ ∈ ℝ^{T × 256}
        └────────────┬───────────────┘
                     ▼
        Hadamard merge:  feat = c[:, None, :] ⊙ φ[None, :, :]     # ℝ^{B × T × 256}
                     ▼
        out_mlp(feat) + out_bias  →  â ∈ ℝ^{B × T × 7}
```

Key design points (as implemented in `deeponet_head_v2.py`):
- **Pooling:** `CrossAttnPool` — 8 learned latent queries, 3 Perceiver-style cross-attention blocks over `memory`.
- **Branch:** MLP → **p = 256** coefficients (the operator latent dimension — basis count, *not* a time discretisation).
- **Trunk:** the chunk index `τ` is lifted by **n = 6 Fourier features** then an MLP to a `τ`-indexed basis `φ(τ) ∈ ℝ^{T×256}`.
- **Merge:** **element-wise (Hadamard) product** `c ⊙ φ` followed by `out_mlp(·) + out_bias` — *not* a plain
  inner product — giving the head a learned, per-coordinate combination of branch coefficients and trunk basis.
- Single forward pass: **no autoregression, no diffusion/flow denoising**.

### 4.3 Persistent-Homology (PH) loss — `ACT + DeepONet + PH` (0 parameters)
A **gated topological regulariser** that matches the persistence diagram (0-dim homology / topological shape) of
the predicted vs ground-truth action trajectory, on top of L1. It is **gated**:
- **warm-up 5000 steps** (off until the policy is past noise),
- **per-sample trigger 0.15** (only fires on samples whose L1 exceeds the threshold; ~8.4 % of samples),
- weight **λ = 0.02**, `ph_k = 8`.

It adds **no parameters** (loss-only). In this study PH did not improve in-distribution accuracy.

---

## 5. Training objective

```
 L = L1(â, a)                      # action-chunk reconstruction (primary)
   + β · KL(q(z|a,s) ‖ N(0,I))     # CVAE style-latent regulariser
   + λ · PH(â, a)                  # gated persistent-homology  (ACT+DeepONet+PH only)
```

---

## 6. Configuration & parameter budget

| Setting | Value |
|---|---|
| `d_model` | 512 |
| `dim_feedforward` | 3200 |
| Encoder layers | 4 |
| Decoder layers (baseline) | 7 |
| Chunk size `T` | 100 |
| Action dim | 7 (6-DoF Δeef + gripper) |
| CVAE latent | 32-d |
| Branch coeffs `p` (DeepONet) | 256 |
| Trunk Fourier features `n` | 6 |
| CrossAttnPool | 8 queries × 3 blocks |
| Optimiser | AdamW, lr 1e-4 (backbone 1e-5), EMA 0.999 |
| Training | 30 000 steps, batch 64, bf16 autocast |
| Inference | receding horizon, replan `k` = 5, EMA weights |

| Variant | Backbone | Action head | **Total** |
|---|---|---|---|
| **ACT** (baseline) | 11.2 M | decoder **37.8 M** | **88.3 M** |
| **ACT + DeepONet** | 11.2 M | DeepONet **10.2 M** | **60.7 M** |
| **ACT + DeepONet + PH** | 11.2 M | DeepONet 10.2 M (+0 PH) | **60.7 M** |

The DeepONet head is a **3.7× smaller action head** (37.8 M → 10.2 M) and ~31 % fewer total parameters, while
matching/beating the decoder in-distribution on all four LIBERO suites (see [`README.md`](README.md)).

---

## 7. Tensor flow (shapes)

```
 images  [B,2,3,256,256] ─ResNet18─►  visual tokens [B, Nv, 512]
 state   [B,8]           ─proj─────►  state token   [B, 1, 512]
 text                    ─LangEnc──►  lang token    [B, 1, 512]
 z       [B,32]          ─proj─────►  z token        [B, 1, 512]   (train: from CVAE enc; test: 0)
                                         │
            concat ─► tokens [B, N, 512] ─► Encoder(4L) ─► memory [B, N, 512]
                                         │
   baseline:  memory ─► Decoder(7L, T queries) ─────────────► â [B, 100, 7]
   deeponet:  memory ─► CrossAttnPool(8q,3blk) ─► branch c[B,256]
                                                  trunk  φ[100,256]
                                       c⊙φ ─► out_mlp ─────────► â [B, 100, 7]
```
