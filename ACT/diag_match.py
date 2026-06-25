import json, torch, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import evaluate_plus_act as P

DS, SUITE = "lerobot/libero_object_image", "libero_object"
meta = LeRobotDatasetMetadata(DS)
bench, tasks = P.list_perturbed_tasks(SUITE)
cfg = json.load(open("act_results/Object/runs/eval_rerun_plus/robustness_plus.json"))["_config"]
pick = []
for c, idxs in cfg["sampled_indices"].items(): pick += idxs[:2]   # 14 tasks across categories

st = meta.stats["observation.images.image"]
def to3(a):
    a = np.asarray(a, dtype=np.float32).squeeze()
    if a.shape == (3,): return a
    if a.ndim == 3 and a.shape[0] == 3: return a.reshape(3, -1).mean(1)
    if a.ndim == 3 and a.shape[2] == 3: return a.reshape(-1, 3).mean(0)
    return a.reshape(-1)[:3]
tm, ts = to3(st["mean"]), to3(st["std"])
if tm.max() > 1.5: tm, ts = tm / 255.0, ts / 255.0      # normalize to [0,1] if needed
tmean = torch.tensor(tm).view(1, 3, 1, 1); tstd = torch.tensor(ts).view(1, 3, 1, 1)
print("train img per-channel mean:", [round(x, 3) for x in tm.tolist()], "std:", [round(x, 3) for x in ts.tolist()])

def match(img):   # affine per-channel match of img(1,3,H,W)[0,1] to training mean/std
    m = img.mean(dim=(0, 2, 3), keepdim=True); s = img.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    return ((img - m) / s * tstd + tmean).clamp(0, 1)

pol, pre, post = P.load_policy("act_results/Object/runs/act_deeponet/checkpoints/LATEST", meta, 5)

def run(correct):
    wins = []
    for idx in pick:
        env = P.LiberoPlusEnv(bench, idx, img_size=256)
        obs = env.reset(seed=0); desc = env.task_description; ok = False
        for _ in range(300):
            d = P.plus_obs_to_policy_input(obs, desc)
            if correct:
                d["observation.images.image"] = match(d["observation.images.image"])
                d["observation.images.wrist_image"] = match(d["observation.images.wrist_image"])
            pin = pre(d); pin = {k: (v.to(P.DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a = post(pol.select_action(pin)).to("cpu").float().numpy().reshape(-1)
            obs, r, done, info = env.step(a)
            if env.check_success(): ok = True; break
            if done: break
        env.close(); wins.append(ok)
    return wins

c = run(False); print(f"\nDeepONet  RAW Plus            : {sum(c)}/{len(c)}  {[int(w) for w in c]}")
m = run(True);  print(f"DeepONet  brightness-matched : {sum(m)}/{len(m)}  {[int(w) for w in m]}")
