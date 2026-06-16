#!/usr/bin/env python
"""
compare.py
==========
Parameter-count (per submodule) and inference-latency comparison of the
baseline and PH SmolVLA models on the Blackwell GPU.

Because PH is a TRAINING-TIME-ONLY regularizer, the two models have byte-for-byte
identical inference graphs -- this script measures and *documents* that parity
(parameter counts equal; latency equal within noise). That zero-inference-cost
property is itself a key result.

Latency: 1000 forward passes of the policy's inference step (select_action) at
batch=1, after warmup, with CUDA synchronization.

Outputs results/compare.json and prints a table.

Usage
-----
    python compare.py --baseline outputs/baseline_full/checkpoints/LATEST \
                      --ph       outputs/ph_full/checkpoints/LATEST
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from modeling_smolvla_ph import SmolVLAPHPolicy, adapt_policy_features_to_dataset
from evaluate import resolve_latest, REPO_DATA, DEV

# Submodule groups for the parameter breakdown.
SUBMODULES = {
    "backbone_vlm (SmolVLM2)": "model.vlm_with_expert.vlm.",
    "action_expert (lm_expert)": "model.vlm_with_expert.lm_expert.",
    "state_proj": "model.state_proj.",
    "action_in_proj": "model.action_in_proj.",
    "action_out_proj": "model.action_out_proj.",
    "action_time_mlp": "model.action_time_mlp",
}


def param_breakdown(policy):
    total = 0
    groups = {k: 0 for k in SUBMODULES}
    other = 0
    for name, p in policy.named_parameters():
        n = p.numel()
        total += n
        matched = False
        for label, prefix in SUBMODULES.items():
            if name.startswith(prefix):
                groups[label] += n
                matched = True
                break
        if not matched:
            other += n
    groups["other"] = other
    groups["TOTAL"] = total
    return groups


def make_dummy_obs(meta):
    """A representative single-frame observation for latency timing."""
    return {
        "observation.images.image": torch.rand(1, 3, 256, 256),
        "observation.images.wrist_image": torch.rand(1, 3, 256, 256),
        "observation.state": torch.zeros(1, 8),
        "task": ["pick up the object and place it"],
    }


@torch.no_grad()
def measure_latency(policy, pre, post, meta, n=1000, warmup=50):
    policy.eval()
    lat = []
    for i in range(n + warmup):
        obs = make_dummy_obs(meta)
        pin = pre(obs)
        pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        # select_action only runs the heavy denoising forward when its action
        # queue is empty; reset each iter to time the full inference step.
        policy.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(pin)
        _ = post(action)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        if i >= warmup:
            lat.append(dt)
    lat = np.array(lat)
    return {
        "mean_ms": float(lat.mean()),
        "std_ms": float(lat.std()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p99_ms": float(np.percentile(lat, 99)),
        "n": int(len(lat)),
    }


def load(ckpt, ph, meta):
    policy = SmolVLAPHPolicy.from_pretrained(resolve_latest(ckpt), ph_enabled=ph).to(DEV).eval()
    pre, post = make_smolvla_pre_post_processors(policy.config, dataset_stats=meta.stats)
    return policy, pre, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ph", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--dataset", default=REPO_DATA)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    meta = LeRobotDatasetMetadata(args.dataset)

    res = {"gpu": torch.cuda.get_device_name(0), "params": {}, "latency": {}}
    for mname, ckpt, ph in [("baseline", args.baseline, False), ("ph", args.ph, True)]:
        policy, pre, post = load(ckpt, ph, meta)
        res["params"][mname] = param_breakdown(policy)
        print(f"[compare] timing {mname} ({args.n} fwd passes, batch=1) ...", flush=True)
        res["latency"][mname] = measure_latency(policy, pre, post, meta, n=args.n)
        del policy
        torch.cuda.empty_cache()

    Path(args.out, "compare.json").write_text(json.dumps(res, indent=2))

    # ---- print table ----
    print("\n=== PARAMETER COUNT (millions) ===")
    labels = list(SUBMODULES) + ["other", "TOTAL"]
    print(f"{'submodule':28s} {'baseline':>12s} {'ph':>12s}")
    for lab in labels:
        b = res["params"]["baseline"][lab] / 1e6
        p = res["params"]["ph"][lab] / 1e6
        print(f"{lab:28s} {b:12.3f} {p:12.3f}")
    print("\n=== INFERENCE LATENCY (ms, batch=1) ===")
    print(f"{'model':10s} {'mean':>8s} {'std':>8s} {'p50':>8s} {'p90':>8s} {'p99':>8s}")
    for m in ("baseline", "ph"):
        L = res["latency"][m]
        print(f"{m:10s} {L['mean_ms']:8.2f} {L['std_ms']:8.2f} {L['p50_ms']:8.2f} "
              f"{L['p90_ms']:8.2f} {L['p99_ms']:8.2f}")
    print(f"\n[compare] saved -> {Path(args.out,'compare.json')}")


if __name__ == "__main__":
    main()
