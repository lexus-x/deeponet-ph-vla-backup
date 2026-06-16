#!/usr/bin/env python
"""
task5_fix_experiment.py
=======================
LIBERO-Spatial task5 ("pick up the black bowl ON the ramekin") is the worst task
for every model (flow 5%). Diagnosis from rollout videos: the gripper servos to a
table-level pose and ends up beside the *elevated* (stacked-on-ramekin) bowl, never
grasping it; the episode then times out.

Hypothesis: the failure is last-mile grasp imprecision under open-loop execution.
Lever: the receding-horizon replan interval (config.n_action_steps). Smaller = more
reactive closed-loop -> should servo onto the raised bowl better.

This sweeps replan in {5(baseline),2,1} for the *trained flow checkpoint* on task5
only, reusing evaluate.py's exact rollout machinery (same seeds 1000+ep) so numbers
are directly comparable to the reported 5%. No retraining. Outputs to a separate
ablation folder.
"""
import os, sys, json, time
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import evaluate as E

CKPT      = "../Spatial/runs/flow_s0/checkpoints/LATEST"
TASK_ID   = 5
N_EP      = int(os.environ.get("N_EP", 12))      # seeds 1000..1000+N_EP-1
MAX_STEPS = 520
REPLANS   = [int(x) for x in os.environ.get("REPLANS", "5,2,1").split(",")]
OUT_DIR   = "../Spatial/ablation_task5"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[t5] loading dataset stats ...", flush=True)
meta = LeRobotDatasetMetadata("lerobot/libero_spatial_image")
stats = meta.stats

results = {}
for replan in REPLANS:
    t0 = time.time()
    policy, pre, post = E.load_policy("flow", CKPT, stats, replan)
    env = E._make_base_libero_env(TASK_ID, suite_name="libero_spatial")
    desc = env.task_description
    succ = []
    for ep in range(N_EP):
        ok = E.rollout(policy, pre, post, env, desc, MAX_STEPS, seed=1000 + ep)
        succ.append(bool(ok))
        print(f"[t5] replan={replan} ep{ep:02d} seed{1000+ep} -> {'PASS' if ok else 'FAIL'}", flush=True)
    env.close()
    rate = float(np.mean(succ))
    results[f"replan_{replan}"] = {"passes": int(sum(succ)), "n": N_EP,
                                   "success_rate": rate, "wall_s": round(time.time()-t0, 1)}
    print(f"[t5] === replan={replan}: {sum(succ)}/{N_EP} = {rate*100:.1f}%  "
          f"({results[f'replan_{replan}']['wall_s']}s) ===", flush=True)
    json.dump({"task": desc, "ckpt": CKPT, "results": results},
              open(os.path.join(OUT_DIR, "replan_sweep.json"), "w"), indent=2)

print("[t5] DONE", json.dumps(results), flush=True)
