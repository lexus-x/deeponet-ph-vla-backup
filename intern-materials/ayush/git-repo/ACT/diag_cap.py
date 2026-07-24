import numpy as np
from PIL import Image
import evaluate_plus_act as P
OUT="/tmp/claude-1000/-home-user-Desktop-Ayush-PH-test/20b8ae9f-eeb6-4e70-8000-90d6cc07d17f/scratchpad"
bench, tasks = P.list_perturbed_tasks("libero_object")
bycat = {}
for t in tasks: bycat.setdefault(t["category"], []).append(t)
for cat, fn in [("Language Instructions", "plus_lang"), ("Robot Initial States", "plus_robot")]:
    t = bycat[cat][0]; env = P.LiberoPlusEnv(bench, t["index"], img_size=256)
    obs = env.reset(seed=0)
    a = np.asarray(obs["agentview_image"]).astype(np.uint8)   # raw, as model receives pre-flip
    Image.fromarray(np.flipud(np.fliplr(a))).save(f"{OUT}/{fn}.png")  # 180-flip = model input
    print(f"{cat}: saved {fn}.png  shape={a.shape} mean={a.mean():.1f}  task={env.task_description}")
    env.close()
