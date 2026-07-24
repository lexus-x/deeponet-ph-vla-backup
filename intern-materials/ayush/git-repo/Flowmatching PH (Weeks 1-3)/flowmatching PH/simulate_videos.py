#!/usr/bin/env python
"""
simulate_videos.py
==================
Render real MuJoCo rollouts to MP4 (H.264, 256x256, 20 FPS, text overlays).

Outputs (under --out, default videos/):
  * LIBERO-10 : 10 tasks x 2 models                       = 20 single videos
  * LIBERO-V  : 4 tasks x 3 perturbation types x 2 models  = 24 single videos
  * Side-by-side baseline|PH comparison videos:
        4 LIBERO-10 tasks + (3 perturbation types on 1 task) = 7 comparison videos
  -> ~51 MP4 files total.

Each frame is the actual observation the policy receives (so LIBERO-V
perturbations are visible), upsampled for legible overlays. Text overlay shows
model, task, perturbation, step, and SUCCESS/RUNNING.

Usage
-----
    MUJOCO_GL=egl python simulate_videos.py \
        --baseline outputs/baseline_full/checkpoints/LATEST \
        --ph       outputs/ph_full/checkpoints/LATEST
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import cv2

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from libero_v_wrapper import make_libero_v_env, _make_base_libero_env
from evaluate import load_policy, env_obs_to_policy_input, REPO_DATA, DEV

FPS = 20
RENDER = 256
UP = 2  # upscale factor for legible text
PERTURBATIONS = ("viewpoint", "lighting", "sensor_noise")


def _two_view(obs):
    """Stack the two camera views the policy uses: agentview | wrist, with a
    thin separator and small per-view labels."""
    a = np.asarray(obs["pixels"]["image"]).astype(np.uint8)       # agentview
    w = np.asarray(obs["pixels"]["image2"]).astype(np.uint8)      # wrist cam
    sep = np.full((a.shape[0], 4, 3), 255, np.uint8)
    return np.concatenate([a, sep, w], axis=1)                    # (256, 516, 3)


def _overlay(frame_hwc_uint8, lines):
    """Upscale (preserving aspect) + draw text lines (list of (text, color))."""
    h, w = frame_hwc_uint8.shape[:2]
    img = cv2.resize(frame_hwc_uint8, (w * UP, h * UP), interpolation=cv2.INTER_NEAREST)
    # camera labels
    cv2.putText(img, "agentview", (8, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(img, "wrist", (img.shape[1]//2+8, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1, cv2.LINE_AA)
    y = 22
    for text, color in lines:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 22
    return img


@torch.no_grad()
def record_rollout(policy, pre, post, env, task_desc, max_steps, seed,
                   label_lines_fn):
    """Run a rollout, returning (frames, success)."""
    policy.reset()
    obs, info = env.reset(seed=seed)
    frames, success = [], False
    for step in range(max_steps):
        view = _two_view(obs)  # agentview | wrist (the 2 viewpoints the policy uses)
        pin = env_obs_to_policy_input(obs, task_desc)
        pin = pre(pin)
        pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(pin)
        action = post(action)
        a = action.to("cpu").float().numpy().reshape(-1)
        obs, reward, terminated, truncated, info = env.step(a)
        success = success or bool(info.get("is_success", False))
        frames.append(_overlay(view, label_lines_fn(step, success)))
        if success or terminated or truncated:
            # hold last frame briefly
            for _ in range(int(FPS * 0.7)):
                frames.append(frames[-1])
            break
    return frames, success


def write_mp4(path, frames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=FPS, codec="libx264",
                     quality=8, macro_block_size=None,
                     ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"[video] wrote {path} ({len(frames)} frames)", flush=True)


def side_by_side(frames_a, frames_b):
    n = max(len(frames_a), len(frames_b))
    out = []
    for i in range(n):
        fa = frames_a[min(i, len(frames_a) - 1)]
        fb = frames_b[min(i, len(frames_b) - 1)]
        h = max(fa.shape[0], fb.shape[0])
        sep = np.zeros((h, 6, 3), np.uint8)
        sep[:] = (255, 255, 255)
        out.append(np.concatenate([fa, sep, fb], axis=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ph", required=True)
    ap.add_argument("--out", default="videos")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--dataset", default=REPO_DATA)
    ap.add_argument("--stats_path", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    meta = LeRobotDatasetMetadata(args.dataset)
    import torch as _torch
    norm_stats = _torch.load(args.stats_path) if args.stats_path else meta.stats
    models = {
        "baseline": load_policy(args.baseline, ph=False, dataset_stats=norm_stats),
        "ph": load_policy(args.ph, ph=True, dataset_stats=norm_stats),
    }
    GREEN, WHITE = (0, 200, 0), (255, 255, 255)

    def lines(model, task, pert, step, success):
        st = ("SUCCESS", GREEN) if success else (f"step {step}", WHITE)
        ls = [(f"{model}", WHITE), (st[0], st[1])]
        if pert:
            ls.insert(1, (f"perturb: {pert}", (0, 180, 255)))
        ls.append((task[:34], WHITE))
        return ls

    # ---- LIBERO-10: 10 tasks x 2 models (single) + 4 comparison ----
    cmp10_frames = {}
    for task_id in range(10):
        for mname, (policy, pre, post) in models.items():
            env = _make_base_libero_env(task_id, suite_name=args.suite)
            td = env.task_description
            fr, succ = record_rollout(
                policy, pre, post, env, td, args.max_steps, args.seed,
                lambda s, ok, m=mname, t=td: lines(m, t, None, s, ok))
            env.close()
            write_mp4(out / "indist" / f"task{task_id:02d}_{mname}.mp4", fr)
            cmp10_frames.setdefault(task_id, {})[mname] = fr
        if task_id < 4:  # 4 side-by-side comparisons
            sb = side_by_side(cmp10_frames[task_id]["baseline"], cmp10_frames[task_id]["ph"])
            write_mp4(out / "comparison" / f"indist_task{task_id:02d}_baseline_vs_ph.mp4", sb)

    # ---- LIBERO-V: 4 tasks x 3 perturb x 2 models (single) ----
    cmpV_frames = {}
    for task_id in range(4):
        for pert in PERTURBATIONS:
            for mname, (policy, pre, post) in models.items():
                env = make_libero_v_env(task_id, perturbation=pert, seed=task_id,
                                        suite_name=args.suite)
                td = env.env.task_description
                fr, succ = record_rollout(
                    policy, pre, post, env, td, args.max_steps, args.seed + 1,
                    lambda s, ok, m=mname, t=td, p=pert: lines(m, t, p, s, ok))
                env.close()
                write_mp4(out / "perturbed" / f"task{task_id:02d}_{pert}_{mname}.mp4", fr)
                if task_id == 0:
                    cmpV_frames.setdefault(pert, {})[mname] = fr
        # 3 comparison videos (per perturbation type) on task 0
        if task_id == 0:
            for pert in PERTURBATIONS:
                sb = side_by_side(cmpV_frames[pert]["baseline"], cmpV_frames[pert]["ph"])
                write_mp4(out / "comparison" / f"perturbed_{pert}_baseline_vs_ph.mp4", sb)

    print("[video] DONE", flush=True)


if __name__ == "__main__":
    main()
