"""Latency microbenchmark for the three ACT variants.
Measures, on this GPU, batch=1, bf16 (matching evaluate_plus_act.py):
  - plan_ms: full forward pass that predicts an action chunk (n_action_steps=1 -> forward every call)
  - amort_ms: amortized per-step latency under receding horizon replan=5 (forward every 5th step)
Writes act_results/latency.json
"""
import json, time, statistics as st, sys, types
from pathlib import Path
import numpy as np
import torch

# Compat shim: installed lerobot moved dataset_to_policy_features to datasets.utils,
# but act_common imports it from datasets.feature_utils (older layout).
try:
    import lerobot.datasets.feature_utils  # noqa
except ModuleNotFoundError:
    from lerobot.datasets.utils import dataset_to_policy_features
    _m = types.ModuleType("lerobot.datasets.feature_utils")
    _m.dataset_to_policy_features = dataset_to_policy_features
    sys.modules["lerobot.datasets.feature_utils"] = _m

from act_common import load_ckpt
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.processor_act import make_act_pre_post_processors

DEV = "cuda"

# inlined from evaluate_plus_act.py (avoid importing it -> pulls in robosuite/numba env)
def _quat2axisangle(quat):
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

DATASET = "lerobot/libero_spatial_image"
CKPT = {
    "act":             "act_results/Spatial/runs/act/checkpoints/30000",
    "act_deeponet":    "act_results/Spatial/runs/act_deeponet/checkpoints/30000",
    "act_deeponet_ph": "act_results/Spatial/runs/act_deeponet_ph/checkpoints/30000",
}
N_WARM, N_ITER = 20, 200

def dummy_obs():
    return {
        "agentview_image": (np.random.rand(256, 256, 3) * 255).astype(np.uint8),
        "robot0_eye_in_hand_image": (np.random.rand(256, 256, 3) * 255).astype(np.uint8),
        "robot0_eef_pos": np.zeros(3, np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32),
        "robot0_gripper_qpos": np.zeros(2, np.float32),
    }

def make_input(pre, task="pick up the black bowl and place it on the plate"):
    pin = pre(plus_obs_to_policy_input(dummy_obs(), task))
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}

@torch.no_grad()
def time_calls(policy, pre, replan, n_warm, n_iter):
    policy.config.n_action_steps = replan
    policy.reset()
    # warmup
    for _ in range(n_warm):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy.select_action(make_input(pre))
    torch.cuda.synchronize()
    per = []
    for _ in range(n_iter):
        pin = make_input(pre)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            policy.select_action(pin)
        torch.cuda.synchronize()
        per.append((time.perf_counter() - t0) * 1000.0)
    return per

def summ(xs):
    xs_s = sorted(xs)
    return {
        "mean_ms": round(st.mean(xs), 3),
        "std_ms": round(st.pstdev(xs), 3),
        "p50_ms": round(xs_s[len(xs)//2], 3),
        "p90_ms": round(xs_s[int(len(xs)*0.9)], 3),
        "min_ms": round(min(xs), 3),
    }

def main():
    meta = LeRobotDatasetMetadata(DATASET)
    gpu = torch.cuda.get_device_name(0)
    out = {"_meta": {"gpu": gpu, "dtype": "bfloat16", "batch": 1, "dataset": DATASET,
                     "n_warm": N_WARM, "n_iter": N_ITER, "chunk_size": 100}}
    for name, ck in CKPT.items():
        policy, variant = load_ckpt(meta, ck)
        policy = policy.to(DEV).eval()
        pre, post = make_act_pre_post_processors(policy.config, dataset_stats=meta.stats)
        n_param = sum(p.numel() for p in policy.parameters())
        plan = time_calls(policy, pre, replan=1, n_warm=N_WARM, n_iter=N_ITER)   # forward every call
        amort = time_calls(policy, pre, replan=5, n_warm=N_WARM, n_iter=N_ITER)  # forward every 5th
        out[name] = {
            "params_M": round(n_param / 1e6, 3),
            "plan_forward": summ(plan),       # cost of one chunk prediction
            "amortized_replan5": summ(amort), # avg per control step at replan=5
            "control_freq_hz_plan": round(1000.0 / st.mean(plan), 1),
            "control_freq_hz_amort": round(1000.0 / st.mean(amort), 1),
        }
        print(f"{name:18} params={out[name]['params_M']:.1f}M  "
              f"plan={out[name]['plan_forward']['mean_ms']:.2f}ms  "
              f"amort@5={out[name]['amortized_replan5']['mean_ms']:.2f}ms  "
              f"({out[name]['control_freq_hz_amort']} Hz)")
        del policy; torch.cuda.empty_cache()
    Path("act_results/latency.json").write_text(json.dumps(out, indent=2))
    print("wrote act_results/latency.json")

if __name__ == "__main__":
    main()
