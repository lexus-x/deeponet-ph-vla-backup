#!/usr/bin/env python
"""
evaluate.py
===========
Closed-loop success-rate evaluation of SmolVLA(+PH) on:

  * LIBERO-10            : 10 tasks x 12 episodes x 2 models   (in-distribution)
  * LIBERO-V (visual)    : 10 tasks x 3 perturbations x 8 episodes x 2 models

Rollouts reuse lerobot's own LiberoProcessorStep (state flattening + 180-deg
image flip) and the SmolVLA pre/post processors, so the eval observation path is
identical to what the policy saw in training. The same single-env rollout
function is used for clean and perturbed envs (LIBERO-V wrapper emits the same
observation format as the base LiberoEnv).

Results are written incrementally to JSON (resumable): per-task and average
success rates per model and benchmark.

Usage
-----
    MUJOCO_GL=egl python evaluate.py \
        --baseline outputs/baseline_full/checkpoints/LATEST \
        --ph       outputs/ph_full/checkpoints/LATEST \
        --out results/

    # fast subset to validate the pipeline:
    MUJOCO_GL=egl python evaluate.py ... --libero10_episodes 2 --liberov_episodes 1 --max_steps 200
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from modeling_smolvla_ph import SmolVLAPHPolicy
from libero_v_wrapper import make_libero_v_env, _make_base_libero_env

DEV = "cuda"
REPO_DATA = "lerobot/libero_10_image"
PERTURBATIONS = ("viewpoint", "lighting", "sensor_noise")
N_TASKS = 10
# LIBERO suite default horizon for the long-horizon "10" suite.
DEFAULT_MAX_STEPS = 520


# --------------------------------------------------------------------------- model
def resolve_latest(ckpt_path: str) -> str:
    """Resolve .../checkpoints/LATEST or .../checkpoints/BEST to a real step dir.
    BEST falls back to LATEST.txt if no BEST.txt was written."""
    p = Path(ckpt_path)
    if p.name in ("LATEST", "BEST"):
        ptr = p.parent / f"{p.name}.txt"
        if not ptr.exists() and p.name == "BEST":
            ptr = p.parent / "LATEST.txt"
        if ptr.exists():
            return str(p.parent / ptr.read_text().strip())
    return str(p)


def load_policy(ckpt: str, ph: bool, dataset_stats):
    ckpt = resolve_latest(ckpt)
    policy = SmolVLAPHPolicy.from_pretrained(ckpt, ph_enabled=ph).to(DEV).eval()
    pre, post = make_smolvla_pre_post_processors(policy.config, dataset_stats=dataset_stats)
    return policy, pre, post


# --------------------------------------------------------------------------- obs conv
def _to_t(x):
    return torch.as_tensor(np.asarray(x)).float()


def _quat2axisangle(quat: torch.Tensor) -> torch.Tensor:
    """(B,4) xyzw quaternion -> (B,3) axis-angle. Mirrors lerobot LiberoProcessorStep."""
    quat = quat.to(torch.float32)
    B = quat.shape[0]
    w = quat[:, 3].clamp(-1.0, 1.0)
    den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    out = torch.zeros((B, 3))
    mask = den > 1e-10
    if mask.any():
        angle = 2.0 * torch.acos(w[mask])
        axis = quat[mask, :3] / den[mask].unsqueeze(1)
        out[mask] = axis * angle.unsqueeze(1)
    return out


def env_obs_to_policy_input(obs, task_description):
    """LiberoEnv obs ({'pixels':{image,image2}, 'robot_state':{...}}) -> policy
    batch, replicating lerobot's LiberoProcessorStep exactly:
      * images -> (1,3,H,W) float[0,1] then 180-deg flip (flip H and W)
      * state  -> [eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)] = (1,8)
    Keys match the dataset/policy (observation.images.image / .wrist_image)."""
    pix = obs["pixels"]
    rs = obs["robot_state"]

    def img(a):  # HWC uint8 -> (1,3,H,W) float[0,1], 180-deg flip
        t = torch.as_tensor(np.asarray(a)).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        return torch.flip(t, dims=[2, 3])

    eef_pos = _to_t(rs["eef"]["pos"]).reshape(1, 3)
    eef_quat = _to_t(rs["eef"]["quat"]).reshape(1, 4)
    grip = _to_t(rs["gripper"]["qpos"]).reshape(1, 2)
    state = torch.cat([eef_pos, _quat2axisangle(eef_quat), grip], dim=-1)  # (1,8)

    return {
        "observation.images.image": img(pix["image"]),
        "observation.images.wrist_image": img(pix["image2"]),
        "observation.state": state,
        "task": [task_description],
    }


# --------------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(policy, pre, post, env, task_description, max_steps, seed):
    policy.reset()
    obs, info = env.reset(seed=seed)
    success = False
    for _ in range(max_steps):
        pin = env_obs_to_policy_input(obs, task_description)
        pin = pre(pin)
        # move tensors to device
        pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(pin)
        action = post(action)
        a = action.to("cpu").float().numpy().reshape(-1)
        obs, reward, terminated, truncated, info = env.step(a)
        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break
    return success


# --------------------------------------------------------------------------- benches
def eval_libero10(models, n_episodes, max_steps, results, out_path,
                  suite="libero_10", bench_name="LIBERO-10"):
    bench = results.setdefault(bench_name, {})
    for mname, (policy, pre, post) in models.items():
        per_model = bench.setdefault(mname, {"per_task": {}, "average": None})
        for task_id in range(N_TASKS):
            key = str(task_id)
            if key in per_model["per_task"]:
                continue  # resume
            env = _make_base_libero_env(task_id, suite_name=suite)
            task_desc = env.task_description
            succ = []
            for ep in range(n_episodes):
                s = rollout(policy, pre, post, env, task_desc, max_steps, seed=1000 + ep)
                succ.append(bool(s))
            env.close()
            rate = float(np.mean(succ))
            per_model["per_task"][key] = {"task": task_desc, "success_rate": rate,
                                          "n": n_episodes}
            print(f"[LIBERO-10] {mname} task{task_id} '{task_desc[:40]}': {rate*100:.1f}%",
                  flush=True)
            _save(results, out_path)
        rates = [v["success_rate"] for v in per_model["per_task"].values()]
        per_model["average"] = float(np.mean(rates)) if rates else None
        _save(results, out_path)


def eval_liberov(models, n_episodes, max_steps, results, out_path,
                 suite="libero_10", bench_name="LIBERO-V"):
    bench = results.setdefault(bench_name, {})
    for mname, (policy, pre, post) in models.items():
        per_model = bench.setdefault(mname, {})
        for pert in PERTURBATIONS:
            ptab = per_model.setdefault(pert, {"per_task": {}, "average": None})
            for task_id in range(N_TASKS):
                key = str(task_id)
                if key in ptab["per_task"]:
                    continue
                env = make_libero_v_env(task_id, perturbation=pert, seed=task_id,
                                        suite_name=suite)
                task_desc = env.env.task_description
                succ = []
                for ep in range(n_episodes):
                    s = rollout(policy, pre, post, env, task_desc, max_steps, seed=2000 + ep)
                    succ.append(bool(s))
                env.close()
                rate = float(np.mean(succ))
                ptab["per_task"][key] = {"task": task_desc, "success_rate": rate,
                                         "n": n_episodes}
                print(f"[LIBERO-V/{pert}] {mname} task{task_id}: {rate*100:.1f}%", flush=True)
                _save(results, out_path)
            rates = [v["success_rate"] for v in ptab["per_task"].values()]
            ptab["average"] = float(np.mean(rates)) if rates else None
            _save(results, out_path)
        # cross-perturbation average
        # average over ALL perturbation nodes present (not just this run's), so
        # splitting perturbations across multiple eval runs stays correct.
        _all_perts = [k for k in ("viewpoint","lighting","sensor_noise")
                      if isinstance(per_model.get(k), dict) and per_model[k].get("average") is not None]
        pavgs = [per_model[k]["average"] for k in _all_perts]
        per_model["average_over_perturbations"] = float(np.mean(pavgs)) if pavgs else None
        _save(results, out_path)


def _save(results, out_path):
    Path(out_path).write_text(json.dumps(results, indent=2))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ph", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--libero10_episodes", type=int, default=12)
    ap.add_argument("--liberov_episodes", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--only", choices=["libero10", "liberov", "both"], default="both")
    ap.add_argument("--suite", default="libero_10", help="LIBERO suite (libero_10, libero_spatial, ...)")
    ap.add_argument("--dataset", default=REPO_DATA, help="dataset repo for normalization stats")
    ap.add_argument("--skip_baseline", action="store_true",
                    help="evaluate only the PH model (baseline already known, e.g. lambda sweeps)")
    ap.add_argument("--stats_path", default=None, help="normalization stats .pt instead of dataset stats")
    ap.add_argument("--perturbations", default=None,
                    help="comma list to restrict LIBERO-V perturbations, e.g. 'viewpoint'")
    args = ap.parse_args()
    global PERTURBATIONS
    if args.perturbations:
        PERTURBATIONS = tuple(p.strip() for p in args.perturbations.split(",") if p.strip())

    # Benchmark labels keyed by suite so spatial + 10 coexist in one JSON.
    indist_name = "LIBERO-10" if args.suite == "libero_10" else args.suite.replace("libero_", "LIBERO-").upper()
    pert_name = f"{indist_name}-V"

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / "success_rates.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    meta = LeRobotDatasetMetadata(args.dataset)
    norm_stats = torch.load(args.stats_path) if args.stats_path else meta.stats

    models = {"ph": load_policy(args.ph, ph=True, dataset_stats=norm_stats)}
    if not args.skip_baseline:
        models["baseline"] = load_policy(args.baseline, ph=False, dataset_stats=norm_stats)
    print(f"[eval] suite={args.suite} loaded models: {list(models)}  out={out_path}", flush=True)

    results.setdefault("_config", {})[args.suite] = {
        "libero10_episodes": args.libero10_episodes,
        "liberov_episodes": args.liberov_episodes,
        "max_steps": args.max_steps,
        "baseline_ckpt": resolve_latest(args.baseline),
        "ph_ckpt": resolve_latest(args.ph),
    }

    if args.only in ("libero10", "both"):
        eval_libero10(models, args.libero10_episodes, args.max_steps,
                      results, out_path, suite=args.suite, bench_name=indist_name)
    if args.only in ("liberov", "both"):
        eval_liberov(models, args.liberov_episodes, args.max_steps,
                     results, out_path, suite=args.suite, bench_name=pert_name)

    # headline summary
    summary = results.get("_summary", {})
    for m in ("baseline", "ph"):
        summary.setdefault(m, {})
        summary[m][f"{indist_name}_avg"] = results.get(indist_name, {}).get(m, {}).get("average")
        summary[m][f"{pert_name}_avg"] = results.get(pert_name, {}).get(m, {}).get("average_over_perturbations")
    results["_summary"] = summary
    _save(results, out_path)
    print("[eval] DONE. summary:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
