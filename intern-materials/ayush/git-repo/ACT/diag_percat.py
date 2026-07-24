import torch, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import evaluate_plus_act as P
meta = LeRobotDatasetMetadata("lerobot/libero_object_image")
bench, tasks = P.list_perturbed_tasks("libero_object")
bycat = {}
for t in tasks: bycat.setdefault(t["category"], []).append(t)
pol, pre, post = P.load_policy("act_results/Object/runs/act_deeponet/checkpoints/LATEST", meta, 5)
print("base env agentview brightness ~101 (reference)\n")
for c in P.CATEGORIES:
    t = bycat[c][0]; env = P.LiberoPlusEnv(bench, t["index"], img_size=256)
    obs = env.reset(seed=0); desc = env.task_description
    bright = np.asarray(obs["agentview_image"]).mean()
    eef0 = np.asarray(obs["robot0_eef_pos"]).copy(); ok = False
    for _ in range(300):
        pin = pre(P.plus_obs_to_policy_input(obs, desc))
        pin = {k: (v.to(P.DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = post(pol.select_action(pin)).to("cpu").float().numpy().reshape(-1)
        obs, r, done, info = env.step(a)
        if env.check_success(): ok = True; break
        if done: break
    moved = np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - eef0)
    print(f"{c:22s} bright={bright:5.1f}  deeponet_ok={ok}  eef_moved={moved:.3f}")
    env.close()
