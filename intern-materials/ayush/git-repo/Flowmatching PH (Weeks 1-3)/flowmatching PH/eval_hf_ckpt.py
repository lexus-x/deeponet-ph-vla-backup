#!/usr/bin/env python
"""
eval_hf_ckpt.py
===============
Evaluate the official `lerobot/smolvla_libero` checkpoint on a LIBERO suite
using lerobot's native eval (eval_policy_all) with its SHIPPED processor, plus
the rename_map (image->camera1, image2->camera2) it was trained with. Renders
videos too. Self-contained so it doesn't disturb the running lambda sweep.

Usage:
  MUJOCO_GL=egl python eval_hf_ckpt.py --policy_path "hugging face ckp/smolvla_libero_local" \
      --task libero_object --n_episodes 50 --out "hugging face ckp/object" --render 8
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv
from lerobot.scripts.lerobot_eval import eval_policy_all, close_envs

RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--task", default="libero_object")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--render", type=int, default=8, help="max episodes rendered to video")
    ap.add_argument("--out", default="hugging face ckp/object")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = LiberoEnv(task=args.task)

    print(f"[hf-eval] loading policy from {args.policy_path}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.policy_path).to("cuda").eval()

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": RENAME},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)

    print(f"[hf-eval] building {args.task} envs (n_envs={args.batch_size})", flush=True)
    envs = make_env(cfg, n_envs=args.batch_size)

    with torch.no_grad(), torch.autocast(device_type="cuda"):
        info = eval_policy_all(
            envs=envs, policy=policy,
            env_preprocessor=env_pre, env_postprocessor=env_post,
            preprocessor=pre, postprocessor=post,
            n_episodes=args.n_episodes, max_episodes_rendered=args.render,
            videos_dir=out / "videos", start_seed=1000,
            max_parallel_tasks=getattr(cfg, "max_parallel_tasks", 1),
        )
    close_envs(envs)

    (out / "eval_info.json").write_text(json.dumps(info, indent=2, default=str))
    print("[hf-eval] OVERALL:", json.dumps(info.get("overall", {}), default=str), flush=True)
    print(f"[hf-eval] saved -> {out}/eval_info.json + videos/", flush=True)


if __name__ == "__main__":
    main()
