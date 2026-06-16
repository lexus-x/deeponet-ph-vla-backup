#!/usr/bin/env python
"""
make_videos.py
==============
Qualitative side-by-side rollout videos of M1 (flow) / M3 (DeepONet) /
M4 (DeepONet+PH) on representative LIBERO-Plus tasks. For each chosen task we run
all three seed-0 models in the SAME perturbed environment, record the agentview
the policy sees, and stack them horizontally with a SUCCESS/FAIL banner.

Uses libero_plus_wrapper (isolated LIBERO-Plus) so there is no import clash with
the original LIBERO. Output: runs/videos/<category>_idx<idx>.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import cv2
import imageio.v2 as imageio

import libero_plus_wrapper as LP  # isolates LIBERO-Plus (sys.path + config + np shim)
from libero_plus_wrapper import LiberoPlusEnv, list_perturbed_tasks
from evaluate_plus import load_policy, plus_obs_to_policy_input

logging.disable(logging.WARNING)
DEV = "cuda"

# (category, difficulty) tasks to illustrate; one easy + a few hard perturbations.
DEFAULT_TASKS = [
    ("Camera Viewpoints", 1),
    ("Camera Viewpoints", 4),
    ("Light Conditions", 4),
    ("Sensor Noise", 3),
]
MODEL_COLORS = {"M1 flow": (80, 80, 255), "M3 DeepONet": (80, 255, 80),
                "M4 DeepONet+PH": (255, 200, 60)}


@torch.no_grad()
def rollout_frames(policy, pre, post, env, task_desc, max_steps):
    policy.reset()
    obs = env.reset(seed=0)
    frames, success = [], False
    for _ in range(max_steps):
        frames.append(np.asarray(obs["agentview_image"])[::-1].copy())  # flip for viewing
        pin = pre(plus_obs_to_policy_input(obs, task_desc))
        pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(pin)
        a = post(action).to("cpu").float().numpy().reshape(-1)
        obs, r, d, i = env.step(a)
        if env.check_success():
            success = True
            frames.append(np.asarray(obs["agentview_image"])[::-1].copy())
            break
        if d:
            break
    return frames, success


def label(frame, text, color, ok):
    f = cv2.resize(frame, (256, 256))
    f = cv2.copyMakeBorder(f, 34, 24, 4, 4, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(f, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    tag = "SUCCESS" if ok else "..."
    tcol = (80, 255, 80) if ok else (180, 180, 180)
    cv2.putText(f, tag, (8, 306), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tcol, 2, cv2.LINE_AA)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", default="runs/m1_flow_s0/checkpoints/LATEST")
    ap.add_argument("--m3", default="runs/m3_deeponet_s0/checkpoints/LATEST")
    ap.add_argument("--m4", default="runs/m4_deeponet_ph_s0/checkpoints/LATEST")
    ap.add_argument("--dataset", default="lerobot/libero_spatial_image")
    ap.add_argument("--out", default="runs/videos")
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=300)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    stats = LeRobotDatasetMetadata(args.dataset).stats

    models = {
        "M1 flow": load_policy("flow", args.m1, stats, args.replan),
        "M3 DeepONet": load_policy("deeponet", args.m3, stats, args.replan),
        "M4 DeepONet+PH": load_policy("deeponet", args.m4, stats, args.replan),
    }
    print("[video] models loaded", flush=True)

    bench, tasks = list_perturbed_tasks("libero_spatial")
    for cat, diff in DEFAULT_TASKS:
        cand = [t for t in tasks if t["category"] == cat and t["difficulty_level"] == diff]
        if not cand:
            continue
        t = cand[0]
        per_model = {}
        for name, (policy, pre, post) in models.items():
            env = LiberoPlusEnv(bench, t["index"], img_size=256)
            fr, ok = rollout_frames(policy, pre, post, env, env.task_description, args.max_steps)
            env.close()
            per_model[name] = (fr, ok)
            print(f"[video] {cat} d{diff} idx{t['index']} | {name}: {'OK' if ok else 'x'} ({len(fr)}f)", flush=True)

        n = max(len(fr) for fr, _ in per_model.values())
        out_frames = []
        for k in range(n):
            cols = []
            for name in models:
                fr, ok = per_model[name]
                f = fr[min(k, len(fr) - 1)]
                cols.append(label(f, name, MODEL_COLORS[name], ok and k >= len(fr) - 2))
                cols.append(np.full((cols[-1].shape[0], 4, 3), 255, np.uint8))
            out_frames.append(np.concatenate(cols[:-1], axis=1))
        name = f"{cat.replace(' ', '_')}_d{diff}_idx{t['index']}.mp4"
        imageio.mimwrite(Path(args.out) / name, out_frames, fps=20, codec="libx264",
                         quality=8, macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"[video] saved {name} ({len(out_frames)} frames)", flush=True)
        for name in models:
            del per_model[name]
    print("[video] DONE ->", args.out, flush=True)


if __name__ == "__main__":
    main()
