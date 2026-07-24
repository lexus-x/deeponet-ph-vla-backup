# Method contribution track (2026-07-24)

## Problem
Measurement of Ayush's M3 is not enough. Need a **named method we authored** that beats his stock baseline.

## Two methods (ours)

### A — POD eigen-motion trunk (matched 30K)
- Swap Fourier trunk for data-derived POD basis (`pod_trunk.py`)
- Train done: `pod_train_spatial_30k/checkpoints/30000`
- Beat target: Spatial in-dist **M3 77%** (same protocol, 10eps×10tasks, replan=5)

### B — Operator-BID (new)
- File: `operator_bid.py` + `--exec bid` in `evaluate_exec.py`
- At each replan: K=8 state-noised candidates; pick min ||a₀ − a_prev|| (backward coherence)
- Not ensembling (averaging failed); not hard pin (failed)
- Beat target: Long stock **m3_r5 60%**, ideally close flow **66%**

## Launch
- POD Spatial eval + Long BID shards on blackwell (parallel)
