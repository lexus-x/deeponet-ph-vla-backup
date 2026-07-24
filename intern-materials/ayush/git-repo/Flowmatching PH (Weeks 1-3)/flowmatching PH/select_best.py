#!/usr/bin/env python
"""
select_best.py
==============
Pick the BEST checkpoint of a trained model by quickly evaluating its last-K
saved checkpoints on a small in-distribution subset, then write BEST.txt.

This is what makes "maximum accuracy" meaningful: more training can overfit and
*reduce* task success, so we select the checkpoint that actually performs best
rather than blindly using the final one.

Usage
-----
  MUJOCO_GL=egl python select_best.py --model_dir output/strong_libero10_baseline \
      --suite libero_10 --dataset lerobot/libero_10_image --ph 0 \
      --n_tasks 4 --n_ep 3 --last_k 4 --max_steps 400
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from modeling_smolvla_ph import SmolVLAPHPolicy
from libero_v_wrapper import _make_base_libero_env
import evaluate as E


def list_checkpoints(model_dir):
    cdir = Path(model_dir) / "checkpoints"
    steps = sorted(int(p.name) for p in cdir.iterdir() if p.name.isdigit())
    return steps


def quick_eval(ckpt_dir, ph, dataset_stats, suite, n_tasks, n_ep, max_steps):
    policy = SmolVLAPHPolicy.from_pretrained(str(ckpt_dir), ph_enabled=ph).to(E.DEV).eval()
    pre, post = make_smolvla_pre_post_processors(policy.config, dataset_stats=dataset_stats)
    succ = []
    for task_id in range(n_tasks):
        env = _make_base_libero_env(task_id, suite_name=suite)
        td = env.task_description
        for ep in range(n_ep):
            s = E.rollout(policy, pre, post, env, td, max_steps, seed=500 + ep)
            succ.append(bool(s))
        env.close()
    del policy
    import torch; torch.cuda.empty_cache()
    return float(np.mean(succ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--dataset", default="lerobot/libero_10_image")
    ap.add_argument("--ph", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=4)
    ap.add_argument("--n_ep", type=int, default=3)
    ap.add_argument("--last_k", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=400)
    args = ap.parse_args()

    meta = LeRobotDatasetMetadata(args.dataset)
    steps = list_checkpoints(args.model_dir)[-args.last_k:]
    print(f"[select] {args.model_dir}: evaluating checkpoints {steps} "
          f"on {args.n_tasks} tasks x {args.n_ep} ep", flush=True)

    best_step, best_rate = None, -1.0
    results = {}
    for st in steps:
        rate = quick_eval(Path(args.model_dir) / "checkpoints" / str(st),
                          bool(args.ph), meta.stats, args.suite,
                          args.n_tasks, args.n_ep, args.max_steps)
        results[st] = rate
        print(f"[select]   ckpt {st}: subset success = {rate*100:.1f}%", flush=True)
        if rate > best_rate:
            best_step, best_rate = st, rate

    best_file = Path(args.model_dir) / "checkpoints" / "BEST.txt"
    best_file.write_text(str(best_step))
    print(f"[select] BEST = {best_step} ({best_rate*100:.1f}%)  -> wrote {best_file}", flush=True)
    # also record the sweep
    (Path(args.model_dir) / "checkpoint_sweep.json").write_text(
        __import__("json").dumps({"subset": results, "best": best_step}, indent=2))


if __name__ == "__main__":
    main()
