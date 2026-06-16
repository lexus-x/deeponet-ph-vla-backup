#!/usr/bin/env python
"""
make_videos_pert.py
===================
Side-by-side rollout videos on a few LIBERO-Plus PERTURBED tasks:
  [ M1 flow | M3 DeepONet-v1 | M3 DeepONet-v2 ]
to illustrate behaviour under camera / lighting / sensor-noise perturbations.
Uses the isolated LIBERO-Plus env (libero_plus_wrapper). Run from parent dir.
Output: runs/videos/pert_<cat>_idx<idx>.mp4
"""
from __future__ import annotations
import os, sys, argparse, logging
from pathlib import Path
import numpy as np, torch, cv2, imageio.v2 as imageio

os.environ.setdefault("MUJOCO_GL", "egl")
logging.disable(logging.WARNING)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "v2"))

import libero_plus_wrapper as LP
from libero_plus_wrapper import LiberoPlusEnv, list_perturbed_tasks
from evaluate_plus import plus_obs_to_policy_input, resolve_latest
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

DEV = "cuda"
TASKS = [("Camera Viewpoints", 3), ("Light Conditions", 3), ("Sensor Noise", 3)]


def load(kind, ckpt):
    ckpt = resolve_latest(ckpt)
    if kind == "flow":
        from modeling_smolvla_ph import SmolVLAPHPolicy as P
        pol = P.from_pretrained(ckpt, ph_enabled=False)
    elif kind == "v1":
        from modeling_smolvla_deeponet import SmolVLADeepONetPolicy as P
        pol = P.from_pretrained(ckpt, ph_enabled=False)
    else:
        from modeling_smolvla_deeponet_v2 import SmolVLADeepONetPolicy as P
        pol = P.from_pretrained(ckpt, ph_enabled=False)
    pol = pol.to(DEV).eval(); pol.config.n_action_steps = 5
    return pol


@torch.no_grad()
def rollout_frames(policy, pre, post, env, desc, max_steps):
    policy.reset(); obs = env.reset(seed=0)
    frames, ok = [], False
    for _ in range(max_steps):
        frames.append(np.asarray(obs["agentview_image"])[::-1].copy())
        pin = pre(plus_obs_to_policy_input(obs, desc))
        pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = post(policy.select_action(pin)).to("cpu").float().numpy().reshape(-1)
        obs, r, d, i = env.step(a)
        if env.check_success():
            ok = True; frames.append(np.asarray(obs["agentview_image"])[::-1].copy()); break
        if d: break
    return frames, ok


def panel(frame, text, color, ok):
    f = cv2.resize(frame, (256, 256))
    f = cv2.copyMakeBorder(f, 34, 24, 4, 4, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(f, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.putText(f, "SUCCESS" if ok else "fail", (8, 306), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (80, 255, 80) if ok else (120, 120, 120), 2, cv2.LINE_AA)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="runs/m1_flow_s0/checkpoints/LATEST")
    ap.add_argument("--v1", default="runs/m3_deeponet_s0/checkpoints/LATEST")
    ap.add_argument("--v2", default="v2/runs/m3v2_s0/checkpoints/LATEST")
    ap.add_argument("--dataset", default="lerobot/libero_spatial_image")
    ap.add_argument("--out", default="runs/videos")
    ap.add_argument("--max_steps", type=int, default=300)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    stats = LeRobotDatasetMetadata(args.dataset).stats

    cols = [("M1 flow", load("flow", args.flow), (80, 80, 255)),
            ("M3 v1", load("v1", args.v1), (200, 150, 255)),
            ("M3 v2", load("v2", args.v2), (80, 255, 80))]
    pres = {name: make_smolvla_pre_post_processors(pol.config, dataset_stats=stats)
            for name, pol, _ in cols}
    bench, tasks = list_perturbed_tasks("libero_spatial")
    print("[vidp] loaded", flush=True)

    for cat, diff in TASKS:
        cand = [t for t in tasks if t["category"] == cat and t["difficulty_level"] == diff]
        if not cand: continue
        t = cand[0]; per = {}
        for name, pol, color in cols:
            env = LiberoPlusEnv(bench, t["index"], img_size=256)
            pre, post = pres[name]
            fr, ok = rollout_frames(pol, pre, post, env, env.task_description, args.max_steps)
            env.close(); per[name] = (fr, ok, color)
            print(f"[vidp] {cat} idx{t['index']} {name}: {'OK' if ok else 'x'}", flush=True)
        n = max(len(fr) for fr, _, _ in per.values())
        frames = []
        for k in range(n):
            row = []
            for name, _, _ in cols:
                fr, ok, color = per[name]
                row.append(panel(fr[min(k, len(fr) - 1)], name, color, ok and k >= len(fr) - 3))
                row.append(np.full((row[-1].shape[0], 4, 3), 255, np.uint8))
            frames.append(np.concatenate(row[:-1], axis=1))
        base = Path(args.out) / f"pert_{cat.replace(' ', '_')}_idx{t['index']}"
        imageio.mimwrite(str(base) + ".mp4", frames, fps=20, codec="libx264", quality=8,
                         macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"[vidp] saved {base.name}.mp4 ({n}f)", flush=True)
    print("[vidp] DONE", flush=True)


if __name__ == "__main__":
    main()
