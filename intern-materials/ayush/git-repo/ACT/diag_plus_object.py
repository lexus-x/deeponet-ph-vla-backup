import torch, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import evaluate_plus_act as P

DS, SUITE = "lerobot/libero_object_image", "libero_object"
meta = LeRobotDatasetMetadata(DS)
bench, tasks = P.list_perturbed_tasks(SUITE)
t = tasks[0]; idx = t["index"]
print("task:", t["name"], "| idx", idx)

def logged(ckpt, tag, steps=250):
    policy, pre, post = P.load_policy(ckpt, meta, 5)
    env = P.LiberoPlusEnv(bench, idx, img_size=256)
    obs = env.reset(seed=0); desc = env.task_description
    eef0 = np.asarray(obs["robot0_eef_pos"]).copy()
    acts = []; succ = False
    for _ in range(steps):
        pin = pre(P.plus_obs_to_policy_input(obs, desc))
        pin = {k: (v.to(P.DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = policy.select_action(pin)
        a = post(a).to("cpu").float().numpy().reshape(-1)
        acts.append(a)
        obs, r, done, info = env.step(a)
        if env.check_success(): succ = True; break
        if done: break
    acts = np.array(acts); eefN = np.asarray(obs["robot0_eef_pos"])
    print(f"\n[{tag}]  steps={len(acts)}  success={succ}")
    print(f"  action mean={np.round(acts.mean(0),3)}")
    print(f"  action range=[{acts.min():.2f},{acts.max():.2f}]  NaN={np.isnan(acts).any()}")
    print(f"  gripper d6: mean={acts[:,6].mean():.3f} range=[{acts[:,6].min():.2f},{acts[:,6].max():.2f}]")
    print(f"  eef start={np.round(eef0,3)} end={np.round(eefN,3)}  moved={np.linalg.norm(eefN-eef0):.4f}m")
    env.close()

logged("act_results/Object/runs/act_deeponet/checkpoints/LATEST", "DeepONet (eval says 0%)")
logged("act_results/Object/runs/act/checkpoints/LATEST", "ACT (eval says 19%)")
