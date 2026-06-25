#!/usr/bin/env python
"""
evaluate_plus.py
================
Closed-loop robustness evaluation of M1/M3/M4 on the LIBERO-Plus benchmark
(sylvestf/LIBERO-plus) over its 7 perturbation dimensions.

* Receding-horizon control: replan every --replan steps (config.n_action_steps),
  identical for all models.
* Per category, a fixed, difficulty-stratified subset of perturbed tasks is
  sampled (same task indices for every model) and run 1 trial each, per the
  LIBERO-Plus convention (num_trials_per_task = 1).
* Self-contained w.r.t. LIBERO: imports libero_plus_wrapper (which isolates the
  LIBERO-Plus package + config) and does NOT import the original-LIBERO eval path.

Models: repeatable  --model NAME=HEAD=CKPT  (HEAD in {flow,deeponet}).
Results stream to <out>/robustness_plus.json (resumable).
"""

from __future__ import annotations

import libero_plus_wrapper as LP  # MUST be imported first (sets sys.path + config)
from libero_plus_wrapper import LiberoPlusEnv, list_perturbed_tasks, CATEGORIES

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

logging.disable(logging.WARNING)
DEV = "cuda"
REPO_DATA = "lerobot/libero_spatial_image"


# --------------------------------------------------------------------------- model
def resolve_latest(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    if p.name in ("LATEST", "BEST"):
        ptr = p.parent / f"{p.name}.txt"
        if not ptr.exists() and p.name == "BEST":
            ptr = p.parent / "LATEST.txt"
        if ptr.exists():
            return str(p.parent / ptr.read_text().strip())
    return str(p)


def load_policy(ckpt: str, meta, replan: int):
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from act_common import load_ckpt
    ckpt = resolve_latest(ckpt)
    policy, variant = load_ckpt(meta, ckpt)
    policy = policy.to(DEV).eval()
    policy.config.n_action_steps = replan
    pre, post = make_act_pre_post_processors(policy.config, dataset_stats=meta.stats)
    return policy, pre, post


# --------------------------------------------------------------------------- obs conv
def _quat2axisangle(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.to(torch.float32)
    w = quat[:, 3].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    out = torch.zeros((quat.shape[0], 3))
    mask = den > 1e-10
    if mask.any():
        angle = 2.0 * torch.acos(w[mask])
        axis = quat[mask, :3] / den[mask].unsqueeze(1)
        out[mask] = axis * angle.unsqueeze(1)
    return out


def plus_obs_to_policy_input(obs, task_description):
    """Raw LIBERO-Plus robosuite obs -> policy batch, matching training convention
    (180-deg image flip; state = [eef_pos(3), quat2axisangle(3), gripper_qpos(2)])."""
    def img(a):
        t = torch.as_tensor(np.asarray(a)).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        return torch.flip(t, dims=[2, 3])

    eef_pos = torch.as_tensor(np.asarray(obs["robot0_eef_pos"])).float().reshape(1, 3)
    eef_quat = torch.as_tensor(np.asarray(obs["robot0_eef_quat"])).float().reshape(1, 4)
    grip = torch.as_tensor(np.asarray(obs["robot0_gripper_qpos"])).float().reshape(1, 2)
    state = torch.cat([eef_pos, _quat2axisangle(eef_quat), grip], dim=-1)
    return {
        "observation.images.image": img(obs["agentview_image"]),
        "observation.images.wrist_image": img(obs["robot0_eye_in_hand_image"]),
        "observation.state": state,
        "task": [task_description],
    }


@torch.no_grad()
def rollout(policy, pre, post, env, task_description, max_steps):
    policy.reset()
    obs = env.reset(seed=0)
    for _ in range(max_steps):
        pin = pre(plus_obs_to_policy_input(obs, task_description))
        pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(pin)
        action = post(action)
        a = action.to("cpu").float().numpy().reshape(-1)
        obs, reward, done, info = env.step(a)
        if env.check_success():
            return True
        if done:
            break
    return False


# --------------------------------------------------------------------------- sampling
def stratified_sample(tasks, n, seed=0):
    """Pick n tasks spread across difficulty levels (deterministic)."""
    import random
    rng = random.Random(seed)
    by_diff = {}
    for t in tasks:
        by_diff.setdefault(t["difficulty_level"], []).append(t)
    for v in by_diff.values():
        v.sort(key=lambda t: t["id"])
        rng.shuffle(v)
    out, levels = [], sorted(by_diff, key=lambda d: (d is None, d))  # None-safe (Goal has None difficulties)
    i = 0
    while len(out) < min(n, len(tasks)):
        lv = levels[i % len(levels)]
        if by_diff[lv]:
            out.append(by_diff[lv].pop())
        i += 1
        if all(len(by_diff[lv]) == 0 for lv in levels):
            break
    return out


def _save(results, out_path):
    Path(out_path).write_text(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="repeatable NAME=HEAD=CKPT (HEAD in flow|deeponet)")
    ap.add_argument("--out", default="runs/eval_plus")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--dataset", default=REPO_DATA)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--n_per_cat", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--categories", default=None, help="comma list to restrict categories")
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()

    cats = CATEGORIES if not args.categories else \
        [c.strip() for c in args.categories.split(",")]

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(args.dataset)
    norm_stats = torch.load(args.stats_path) if args.stats_path else meta.stats

    # enumerate + sample tasks per category (same for all models)
    bench, tasks = list_perturbed_tasks(args.suite)
    by_cat = {c: [t for t in tasks if t["category"] == c] for c in cats}
    sampled = {c: stratified_sample(by_cat[c], args.n_per_cat, seed=42) for c in cats}
    for c in cats:
        print(f"[plus] {c}: sampled {len(sampled[c])} / {len(by_cat[c])} tasks", flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / "robustness_plus.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.setdefault("_config", {}).update({
        "suite": args.suite, "replan": args.replan, "n_per_cat": args.n_per_cat,
        "max_steps": args.max_steps, "categories": cats,
        "sampled_indices": {c: [t["index"] for t in sampled[c]] for c in cats},
    })
    _save(results, out_path)

    models = {}
    for spec in args.model:
        name, ckpt = spec.split("=", 1)
        models[name] = resolve_latest(ckpt)

    for name, ckpt in models.items():
        mres = results.setdefault(name, {})
        policy, pre, post = load_policy(ckpt, meta, args.replan)
        print(f"[plus] loaded {name} <- {ckpt}", flush=True)
        for c in cats:
            cres = mres.setdefault(c, {"per_task": {}, "average": None})
            for t in sampled[c]:
                idx = t["index"]
                key = str(idx)
                if key in cres["per_task"]:
                    continue
                env = LiberoPlusEnv(bench, idx, img_size=args.img_size)
                succ = rollout(policy, pre, post, env, env.task_description, args.max_steps)
                env.close()
                cres["per_task"][key] = {"name": t["name"], "difficulty": t["difficulty_level"],
                                         "success": bool(succ)}
                print(f"[plus/{name}/{c}] idx{idx} d{t['difficulty_level']}: "
                      f"{'OK' if succ else 'x'}", flush=True)
                _save(results, out_path)
            vals = [v["success"] for v in cres["per_task"].values()]
            cres["average"] = float(np.mean(vals)) if vals else None
            _save(results, out_path)
        cat_avgs = [mres[c]["average"] for c in cats if mres.get(c, {}).get("average") is not None]
        mres["robustness_average"] = float(np.mean(cat_avgs)) if cat_avgs else None
        _save(results, out_path)
        print(f"[plus] {name} robustness avg = {mres['robustness_average']}", flush=True)
        del policy
        torch.cuda.empty_cache()

    print("[plus] DONE", flush=True)
    _save(results, out_path)


if __name__ == "__main__":
    main()
