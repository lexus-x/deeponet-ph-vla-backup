#!/usr/bin/env python
"""
compare.py  (DeepONet campaign)
===============================
GPU gate run BEFORE training: compares the flow-matching head and the DeepONet
head on the two axes that are architectural (so they are known before any
accuracy result):

  1. Active parameter count  (backbone / head / total), excluding the dead
     flow-matching expert for the DeepONet model.
  2. End-to-end inference latency: full closed-loop action-chunk prediction
     (VLM prefix + head), averaged over many calls, with policy.reset() each
     call so a fresh chunk is predicted every time.

Both policies are loaded from lerobot/smolvla_base (untrained head) -- latency &
param count do not depend on training, so this is a valid pre-flight gate.

Writes <out>/compare.json and prints a table.
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

DEV = "cuda"


def fake_env_obs():
    """Mimic the LiberoEnv observation dict consumed by env_obs_to_policy_input."""
    return {
        "pixels": {
            "image": (np.random.rand(256, 256, 3) * 255).astype(np.uint8),
            "image2": (np.random.rand(256, 256, 3) * 255).astype(np.uint8),
        },
        "robot_state": {
            "eef": {"pos": np.random.randn(3), "quat": np.random.randn(4)},
            "gripper": {"qpos": np.random.randn(2)},
        },
    }


def count_active(policy):
    is_bb = policy._is_backbone
    is_dead = getattr(policy, "_is_dead", lambda n: False)
    bb = head = dead = 0
    for n, p in policy.named_parameters():
        if is_dead(n):
            dead += p.numel()
        elif is_bb(n):
            bb += p.numel()
        else:
            head += p.numel()
    return {"backbone": bb, "head": head, "dead_excluded": dead, "active_total": bb + head}


@torch.no_grad()
def measure_latency(policy, pre, n=100, warmup=15):
    from evaluate import env_obs_to_policy_input
    pin = pre(env_obs_to_policy_input(fake_env_obs(), "pick up the black bowl"))
    pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
    for _ in range(warmup):
        policy.reset()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy.select_action(pin)
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        policy.reset()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy.select_action(pin)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def load(head, base_ckpt, dataset, stats):
    if head == "flow":
        from modeling_smolvla_ph import SmolVLAPHPolicy, adapt_policy_features_to_dataset
        pol = SmolVLAPHPolicy.from_pretrained(base_ckpt, ph_enabled=False)
    else:
        from modeling_smolvla_deeponet import SmolVLADeepONetPolicy, adapt_policy_features_to_dataset
        pol = SmolVLADeepONetPolicy.from_pretrained(base_ckpt, ph_enabled=False)
    meta = LeRobotDatasetMetadata(dataset)
    adapt_policy_features_to_dataset(pol, meta)
    pol = pol.to(DEV).eval()
    pol.config.n_action_steps = 5
    pre, _ = make_smolvla_pre_post_processors(pol.config, dataset_stats=stats)
    return pol, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", default="lerobot/smolvla_base")
    ap.add_argument("--dataset", default="lerobot/libero_spatial_image")
    ap.add_argument("--out", default="runs/compare")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    meta = LeRobotDatasetMetadata(args.dataset)
    stats = meta.stats

    report = {}
    for head in ("flow", "deeponet"):
        print(f"\n=== loading {head} ===", flush=True)
        pol, pre = load(head, args.base_ckpt, args.dataset, stats)
        params = count_active(pol)
        lat_mean, lat_std = measure_latency(pol, pre, n=args.n)
        report[head] = {"params": params, "latency_ms_mean": lat_mean, "latency_ms_std": lat_std,
                        "num_steps": getattr(pol.config, "num_steps", None)}
        print(f"[{head}] active: backbone={params['backbone']/1e6:.1f}M "
              f"head={params['head']/1e6:.2f}M total={params['active_total']/1e6:.1f}M "
              f"(dead excluded={params['dead_excluded']/1e6:.1f}M)")
        print(f"[{head}] latency = {lat_mean:.1f} +/- {lat_std:.1f} ms / chunk")
        del pol
        torch.cuda.empty_cache()

    f, d = report["flow"], report["deeponet"]
    report["_comparison"] = {
        "head_param_ratio_flow_over_deeponet":
            f["params"]["head"] / max(1, d["params"]["head"]),
        "latency_speedup_flow_over_deeponet":
            f["latency_ms_mean"] / max(1e-9, d["latency_ms_mean"]),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "compare.json").write_text(json.dumps(report, indent=2))

    print("\n================ GATE SUMMARY ================")
    print(f"flow     head params : {f['params']['head']/1e6:7.2f}M | latency {f['latency_ms_mean']:6.1f} ms")
    print(f"deeponet head params : {d['params']['head']/1e6:7.2f}M | latency {d['latency_ms_mean']:6.1f} ms")
    print(f"head size  : flow is {report['_comparison']['head_param_ratio_flow_over_deeponet']:.1f}x larger")
    print(f"latency    : deeponet is {report['_comparison']['latency_speedup_flow_over_deeponet']:.2f}x faster")
    print("=============================================")


if __name__ == "__main__":
    main()
