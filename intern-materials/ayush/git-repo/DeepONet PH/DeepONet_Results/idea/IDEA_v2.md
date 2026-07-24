# DeepONet-v2 action head for SmolVLA — detailed idea

**The DeepONet-v2 action head for SmolVLA.** Modern vision-language-action (VLA)
models such as SmolVLA generate robot motion with an iterative *flow-matching*
(or diffusion) action expert: a large (~100M-parameter) transformer that, at
inference time, must run roughly ten sequential denoising passes to turn random
noise into an executable 50-step action chunk. This is both parameter-heavy and
slow. Our central idea is to replace that iterative expert with a **DeepONet** — a
neural *operator* that learns a direct mapping from the current observation to the
entire future action trajectory in a **single forward pass**. Formally, we treat
action generation as learning an operator G that takes the multimodal observation
(the two camera views, the language instruction, and the proprioceptive robot
state, all encoded by the frozen-then-fine-tuned SmolVLM-2 backbone) and returns a
continuous function a(tau) that gives the commanded action at every normalized
time tau in [0,1] across the chunk horizon. In classic DeepONet fashion this
operator factorizes into two networks whose outputs are combined multiplicatively:
a **branch** network that encodes the input observation into a set of
coefficients, and a **trunk** network that encodes the query time tau into a set
of basis functions; their elementwise product, passed through a small output MLP,
yields the action at that timestep. Because the trunk is a smooth function of tau,
the head is resolution-free and produces the whole chunk at once, eliminating the
ten-step denoising loop entirely.

**What changed from v1 to v2, and why.** Our first DeepONet (v1, ~2.3M
parameters) summarized the entire observation into a *single* mean-pooled context
vector before feeding it to the branch. While extremely small and fast (43x fewer
head parameters and 6.4x lower latency than the flow-matching expert), this
created a severe information bottleneck: averaging all of the visual-language
tokens into one vector blurs away *where* things are in the scene. Empirically
this showed up as a very specific failure pattern — v1 stayed competitive on most
LIBERO-Spatial tasks but **collapsed on spatially-grounded tasks** (e.g. "the bowl
*on the stove*" or "*on the wooden cabinet*"), dragging the in-distribution
average down by ~25 points relative to flow-matching. DeepONet-v2 fixes this at
the root by making the **branch read the full observation token sequence through
cross-attention** instead of a pooled vector. Concretely, a small set of learned
query tokens (K = 8) attend, over three Perceiver-style cross-attention blocks
(model width 512, 8 heads), to *all* of the backbone's prefix tokens, extracting
several focused, spatially-aware context vectors rather than one averaged summary.
This gives the DeepONet branch the same rich, attention-based access to fine visual
detail that the flow-matching expert enjoyed, while remaining within a ~10.4M-
parameter head (still ~10x smaller than the original expert and within the 10-15M
budget). We additionally enrich the trunk with **Fourier positional features** of
tau so it can represent fine temporal structure in the trajectory, and widen the
bilinear basis to p = 256. Crucially, none of this departs from the DeepONet
framework — the branch is simply a stronger encoder of the input function — so the
model remains, end to end, a single-pass operator
a(tau) = OutMLP( branch(observation) (x) trunk(tau) ).

**Training, regularization, and evaluation.** The head is dropped into the
existing SmolVLA pipeline and trained with the same two-stage recipe as the
baselines for a fair comparison: a short stage-1 warm-up of the new head with the
backbone frozen, followed by full-backbone fine-tuning with differential learning
rates, bfloat16 with gradient checkpointing, and an exponential moving average
(decay 0.999) of the weights used for evaluation. The model is supervised by
direct regression (MSE) between its predicted action chunk and the expert
demonstration chunk — there is no velocity field and no noise schedule, only the
operator output. We instantiate two variants: **M3-v2 (DeepONet only)** and
**M4-v2 (DeepONet + Persistent-Homology regularization)**, where the latter adds,
with a small weight lambda = 0.02, a differentiable topological surrogate loss
that matches the sorted pairwise-distance "shape" of the predicted action chunk to
that of the expert chunk, encouraging the predicted trajectory to share the
expert's coarse geometric structure. All models are compared head-to-head against
the flow-matching baseline (M1) under identical data (LIBERO-Spatial, batch 48,
~7.5 epochs), identical closed-loop control (receding-horizon replanning every 5
steps), and across three random seeds for statistical error bars, on both
in-distribution LIBERO-Spatial and the seven-dimensional LIBERO-Plus robustness
benchmark. The hypothesis DeepONet-v2 tests is precise: that restoring spatial
information flow through cross-attention recovers the accuracy lost to the v1
mean-pool bottleneck — closing the gap to flow-matching — *while preserving* the
single-pass DeepONet head's order-of-magnitude advantages in size and inference
latency.
