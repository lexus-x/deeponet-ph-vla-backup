# The Persistent-Homology (PH) loss — in depth

**Source:** `DeepONet PH/v2/ph_loss.py` · **Models:** `M3` = PH off, `M4` = PH on.

## 1. What persistent homology is

**Persistent homology (PH)** is the core tool of **Topological Data Analysis (TDA)**. Given a point set or signal, you
build a growing family of simplicial complexes (a *filtration* — e.g. Vietoris–Rips by connecting points within a
growing radius, or sublevel sets of a signal) and track when **topological features** appear and disappear:

- **H₀** features = connected components (clusters).
- **H₁** features = loops / holes.

Each feature is recorded as a `(birth, death)` pair; the collection is the **persistence diagram**. Features that
persist over a wide range of scales are "real" structure; short-lived ones are noise. PH is a principled, multi-scale
descriptor of the **global shape** of data.

References: Edelsbrunner & Harer, *Computational Topology* (2010); Carlsson, *Topology and Data*, Bull. AMS (2009).

## 2. Why use it as a loss for action trajectories

A robot action chunk is a short **trajectory** in action space. The standard MSE/L1 loss is **point-wise**: it
penalises each timestep independently and is blind to the trajectory's **global geometry** — its overall shape,
whether it loops, how its segments connect.

The **PH loss** adds a topological term: it compares the **persistence diagrams** of the predicted vs ground-truth
trajectory and penalises their difference (e.g. a Wasserstein/bottleneck-style distance between diagrams). This pushes
the policy to produce trajectories whose **global topological shape** matches the demonstration, not just trajectories
that are locally close.

```
total loss = L_regression (MSE/L1, point-wise)  +  λ · L_PH (topological, shape)
```

The training logs show both terms (e.g. `mse=… L1=… PH=…`); for flow/M3 runs `PH=0.0000` because PH is disabled.

## 3. What it buys (empirically, LIBERO-Spatial)

| Model | In-dist | Robustness |
|---|---|---|
| M3 DeepONet (PH off) | 80.3 | **38.5** |
| M4 DeepONet + PH (PH on) | **81.5** | 32.6 |

- **PH improves in-distribution accuracy** (+1.2 points) — the topological regulariser helps fit the demonstrated
  motion shape.
- **PH does *not* improve robustness** — the bare operator (M3) is the most robust. The robustness advantage comes
  from the **operator structure** (Branch⊗Trunk + cross-attn pooling), not from PH.

**Honest conclusion:** PH is best framed as an **accuracy/shape regulariser**, a secondary contribution. The headline
robustness story belongs to the DeepONet operator head itself. Both are reported so the contribution of each is clear.

## 4. Practical notes
- PH requires a **differentiable** persistence computation to backprop through (persistence diagrams are piecewise-
  differentiable w.r.t. the input coordinates). The implementation uses a TDA backend (ripser/persim-style) wrapped to
  return a differentiable diagram distance.
- PH is **computationally heavier** per step than MSE; it is applied to the action trajectory only (low-dimensional),
  keeping cost manageable.
- `λ` (the PH weight) trades off point-wise accuracy vs topological shape; see `ph_loss.py` and the training scripts
  for the value used.
