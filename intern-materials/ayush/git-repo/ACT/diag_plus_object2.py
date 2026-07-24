import json, torch, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import evaluate_plus_act as P

DS, SUITE = "lerobot/libero_object_image", "libero_object"
meta = LeRobotDatasetMetadata(DS)
bench, tasks = P.list_perturbed_tasks(SUITE)
by_idx = {t["index"]: t for t in tasks}

# exact indices the eval sampled (from the re-run JSON), take a slice across categories
cfg = json.load(open("act_results/Object/runs/eval_rerun_plus/robustness_plus.json"))["_config"]
si = cfg["sampled_indices"]
pick = []
for c, idxs in si.items():
    pick += idxs[:2]   # 2 per category -> 14 tasks spanning all categories
print(f"reproducing {len(pick)} eval-sampled Object tasks, both models\n")

def run(ckpt, name):
    policy, pre, post = P.load_policy(ckpt, meta, 5)
    wins = []
    for idx in pick:
        env = P.LiberoPlusEnv(bench, idx, img_size=256)
        obs = env.reset(seed=0); desc = env.task_description
        ok = False
        for _ in range(300):
            pin = pre(P.plus_obs_to_policy_input(obs, desc))
            pin = {k: (v.to(P.DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a = post(policy.select_action(pin)).to("cpu").float().numpy().reshape(-1)
            obs, r, done, info = env.step(a)
            if env.check_success(): ok = True; break
            if done: break
        env.close(); wins.append(ok)
    print(f"{name:12s}: {sum(wins)}/{len(wins)} success  -> {[int(w) for w in wins]}")
    return wins

run("act_results/Object/runs/act/checkpoints/LATEST", "ACT")
run("act_results/Object/runs/act_deeponet/checkpoints/LATEST", "DeepONet")
