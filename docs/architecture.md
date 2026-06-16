# Architecture — before/after, parameters, and cross-model porting

## 1. The generic VLA pattern

Every VLA factorises into a **backbone** (vision+language → prefix tokens) and an **action head/expert**
(prefix tokens → action chunk). Our contribution **only replaces the head**; the backbone is shared and unchanged.

```
 BEFORE (flow/diffusion expert)            AFTER (DeepONet operator head)
 ┌──────────────────────┐                 ┌──────────────────────┐
 │  VLM / encoder         │  (unchanged)    │  VLM / encoder         │ (unchanged)
 └──────────┬───────────┘                 └──────────┬───────────┘
            │ prefix tokens                            │ prefix tokens
            ▼                                           ▼
 ┌──────────────────────┐                 ┌──────────────────────────────────┐
 │ FLOW/DIFFUSION EXPERT  │      ⟶          │ DeepONet HEAD (~10.4 M)            │
 │ large · iterative      │     swap        │ CrossAttnPool→Branch⊗Trunk→OutMLP │
 │ (10-step denoise)      │                 │ single forward pass               │
 └──────────────────────┘                 └──────────────────────────────────┘
```

## 2. SmolVLA — measured parameter breakdown

The DeepONet head (`DeepONet PH/v2/deeponet_head_v2.py`), total **10,366,784 ≈ 10.4 M**:

| Component | Params | Detail |
|---|---|---|
| **CrossAttnPool** | 6.81 M | `in_proj` Linear(960→512) = 0.49 M · 8 learned queries (8×512) · 3× cross-attn block (2.10 M each = attn 1.05 M + FF 1.05 M + LayerNorms) = 6.31 M |
| **Branch** | 3.34 M | Linear(4096→768)=3.15 M + Linear(768→256)=0.20 M (4096 = 8 queries × 512) |
| **Trunk** | 0.14 M | Linear(33→256)→256→256; input 33 = τ(1) + Fourier(32) |
| **Output MLP** | 0.07 M | Linear(256→256)+Linear(256→32) + 32 bias |
| **Total** | **10.4 M** | vs flow expert **99.9 M** |

Whole model: backbone **~350 M** (SmolVLM2, 16 layers) + head. **450 M → 360 M** (−20%) if the dead flow expert is
deleted; in the current code the dead modules are *frozen but retained* (delete for the storage win).

Inference latency (measured): **148 ms → 29.5 ms** per action chunk (~5×), because the operator is a single forward
pass instead of a 10-step flow-matching denoising loop.

## 3. Why the head is ~constant across backbones

The head's size is set by **`d_model = 512`** (cross-attn width) and **`p = 256`** (operator basis), which are our
design choices — **not** by the backbone. Only the **`in_proj`** (backbone hidden → 512) rescales with the backbone's
context dimension:

| Backbone hidden dim | `in_proj` size | Head total |
|---|---|---|
| 960 (SmolVLA) | 0.49 M | 10.4 M |
| ~2048 (PaliGemma / π0) | ~1.05 M | ~11 M |
| ~4096 (large LLM backbones) | ~2.1 M | ~12 M |

So the head is **~10–12 M for any backbone**, and the parameter saving = `(model's native action head) − ~11 M`.

## 4. Cross-model porting table

The DeepONet head can replace the action head/expert of any VLA. Below: native head size, the ~11 M replacement, and
the resulting saving. **Only SmolVLA is measured here**; the rest use published model sizes (approximate, flagged).

| Model | Total | Backbone | Native action head | → DeepONet | Head reduction | Category |
|---|---|---|---|---|---|---|
| **SmolVLA** ✅ measured | 450 M | 350 M | flow expert **99.9 M** | 10.4 M | **−90 M (−90%, 9.6×)** | 🟢 big win |
| **π0** | ~3.3 B | PaliGemma ~3.0 B | action expert ~300 M | ~11 M | **−289 M (−96%)** | 🟢 big win |
| **π0.5** | ~3.4 B | ~3.0 B | ~300 M | ~11 M | ~−290 M (−96%) | 🟢 big win |
| **GR00T N1.5** | ~3 B | Eagle VLM ~2.2 B | DiT head ~0.5–0.7 B *(est)* | ~12 M | ~−0.5 B *(est)* | 🟢 big win |
| **RDT-1B** | 1.2 B | enc ~0.4 B | diffusion transformer ~0.8 B *(est)* | ~11 M | ~−0.8 B *(est)* | 🟢 big win |
| **OpenVLA / -OFT** | 7 B | Llama-7B+enc ~7 B | autoregressive (no expert) / tiny OFT head | ~11 M | adds a head | 🔵 backbone-dominated |
| **ACT** | ~80 M | ResNet+enc ~50 M | transformer decoder ~30 M *(est)* | ~11 M | ~−19 M *(est)* | 🟡 modest |
| **Octo-Base** | 93 M | ViT+transf ~90 M | diffusion head ~3 M | ~11 M | +8 M (increase) | 🟡 not a param win |

**Three categories:**
- 🟢 **Big param + latency win** (π0, π0.5, GR00T, RDT, SmolVLA): replaces a large flow/diffusion expert → 90–96% head reduction + ~5× faster.
- 🟡 **Architectural / robustness fit, not a size win** (Octo, ACT): heads already tiny; value is the operator prior + robustness, not compression. Octo's readout-token design even maps cleanly onto the cross-attention pooler.
- 🔵 **Backbone-dominated** (OpenVLA 7 B): a ~11 M head against a 7 B backbone makes whole-model % ≈ 0; wins are inference/robustness only.

> Estimates marked *(est)* are internal head/backbone splits I could not measure directly — do not cite them as exact.
