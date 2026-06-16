# The DeepONet operator action head — in depth

**Source:** `DeepONet PH/v2/deeponet_head_v2.py`

## 1. Operator learning background

A neural network usually maps **vectors → vectors**. An **operator network** maps **functions → functions**
(or, more practically, a conditioning input → an entire output function). The **DeepONet**
(Lu et al., *Nature Machine Intelligence* 2021, [arXiv:1910.03193](https://arxiv.org/abs/1910.03193)) is the canonical
architecture, backed by a **universal approximation theorem for operators**. It factorises the operator `G` as:

```
G(u)(y) ≈ Σ_{k=1..p}  b_k(u) · t_k(y)
          └── Branch ──┘ └─ Trunk ─┘
```

- **Branch** network encodes the input function/condition `u` into `p` coefficients `b(u) ∈ ℝᵖ`.
- **Trunk** network encodes the query coordinate `y` into `p` basis values `t(y) ∈ ℝᵖ`.
- Their **inner product** gives the output function evaluated at `y`. The Trunk learns a **basis** over the output
  domain; the Branch learns the **coefficients** for a specific input.

## 2. Why this fits a VLA action head

A VLA must output an **action chunk** — a short trajectory `a(τ)`, `τ ∈ [0,1]` the normalised time within the chunk.
That is literally **a function of time conditioned on the observation**:

```
G : observation  ↦  a(·)            a(τ) = action at normalised time τ
```

This is an operator-learning problem. Rather than denoising noise into the chunk (flow/diffusion), we **learn the
operator directly** and evaluate it at the query times we need — in **one forward pass**.

## 3. The head, stage by stage

Input: VLM **prefix tokens** `[B, N, 960]` (+ padding mask). Output: **action chunk** `[B, T, 32]`.

### (a) CrossAttnPool — Perceiver-style pooling
- `in_proj`: Linear(960 → `d_model`=512) projects prefix tokens.
- `K = 8` **learned query tokens** cross-attend over the (variable-length) prefix through **3 cross-attention blocks**
  (8 heads, FF, LayerNorm). Output: a fixed `[B, 8, 512]` summary — independent of prefix length.
- *Why:* compress the whole multimodal context into a compact, fixed observation code for the Branch. Ablations show
  this is the **most important** component (3→1 blocks drops both accuracy and robustness).

### (b) Branch → coefficients `c`
- Flatten the pooled `[B, 8, 512] → [B, 4096]`.
- MLP `4096 → 768 → 256` → coefficients `c ∈ ℝ²⁵⁶` ("what to do" code). `p = 256`.

### (c) Trunk → basis `φ(τ)`
- Query coordinate `τ ∈ [0,1]` (one per timestep in the chunk).
- **Fourier features**: lift τ to `[sin(2πfτ), cos(2πfτ)]` for 16 frequencies → 32 dims, concatenated with τ → 33 dims.
  (Fourier features let an MLP represent high-frequency functions of a low-dim input — Tancik et al., NeurIPS 2020.)
- MLP `33 → 256 → 256 → 256` → basis `φ(τ) ∈ ℝ²⁵⁶`.

### (d) Parameter-free merge + Output MLP
```
a(τ) = OutMLP( c ⊙ φ(τ) )          ⊙ = element-wise (Hadamard) product
```
- The merge has **no parameters** (this is the DeepONet inner-product, here as a Hadamard product feeding a small MLP).
- OutMLP `256 → 256 → 32` maps to the action dimension (32 for SmolVLA).
- Evaluating the Trunk at `τ = 0, 1/T, …, (T-1)/T` and reusing the **same** `c` produces the **whole chunk at once**.

## 4. Clarifying `p`, `d_model`, and τ

- **`p = 256`** is the **operator basis size** (latent dimension) — the number of learned basis functions the Trunk
  produces and the length of the Branch coefficient vector. It is **not** a time discretisation. The chunk is
  continuous in τ; you may query any τ ∈ [0,1].
- **`d_model = 512`** is the internal width of the cross-attention pooler (and its queries) — a transformer hidden
  size, unrelated to `p`.
- **τ** is the normalised position within the action chunk (0 = first action, 1 = last). For a chunk of `T` steps we
  query `T` values of τ.

## 5. Design choices validated by ablations (Spatial, see `docs/results.md`)

| Knob | Setting | Effect of removing/shrinking |
|---|---|---|
| Cross-attn blocks | 3 | 3→1 ⇒ worst (78.2 in-dist, 30.8 robust) — pooling is critical |
| Operator merge | Branch⊗Trunk | plain regression head (no operator) ⇒ higher in-dist (83.0) but −6 robustness (32.7) |
| Basis `p` | 256 | 256→64 ⇒ +in-dist, −robustness |
| Fourier τ | on | off ⇒ ~same robustness, slightly higher in-dist (a minor knob) |

**Takeaway:** the **cross-attention pooler + operator merge** are what deliver robustness; a regression head fits the
training distribution slightly better but loses the operator's out-of-distribution stability.
