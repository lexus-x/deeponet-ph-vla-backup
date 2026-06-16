#!/usr/bin/env python
"""Per-suite rollout videos: [ M1 flow | M3 DeepONet-v2 | M4 DeepONet+PH-v2 ] on
all 10 tasks of a given LIBERO suite (object/goal). Reuses tested helpers from
make_videos_indist. Output: <out>/suite_taskNN.mp4 (+ poster). Run from v2/."""
import argparse, sys, os
from pathlib import Path
import numpy as np, imageio.v2 as imageio
sys.path.insert(0, "..")
from make_videos_indist import load, rollout_frames, panel
from libero_v_wrapper import _make_base_libero_env
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True)        # libero_object / libero_goal
ap.add_argument("--dataset", required=True)      # lerobot/libero_object_image ...
ap.add_argument("--flow", required=True)
ap.add_argument("--m3", required=True)
ap.add_argument("--m4", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_steps", type=int, default=300)
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
a = ap.parse_args()
Path(a.out).mkdir(parents=True, exist_ok=True)
stats = LeRobotDatasetMetadata(a.dataset).stats

cols = [("M1 flow", load("flow", a.flow), (80,80,255)),
        ("M3 v2",   load("v2",   a.m3),   (80,255,80)),
        ("M4 v2+PH", load("v2",  a.m4),   (60,200,255))]
pres = {n: make_smolvla_pre_post_processors(p.config, dataset_stats=stats) for n,p,_ in cols}
print(f"[vid-{a.suite}] 3 policies loaded", flush=True)

for tid in [int(t) for t in a.tasks.split(",")]:
    per = {}
    for name, pol, color in cols:
        env = _make_base_libero_env(tid, suite_name=a.suite)
        pre, post = pres[name]
        fr, ok = rollout_frames(pol, pre, post, env, env.task_description, a.max_steps)
        env.close(); per[name] = (fr, ok, color)
        print(f"[vid-{a.suite}] task{tid} {name}: {'OK' if ok else 'x'}", flush=True)
    n = max(len(fr) for fr,_,_ in per.values()); frames=[]
    for k in range(n):
        row=[]
        for name,_,_ in cols:
            fr,ok,color=per[name]
            row.append(panel(fr[min(k,len(fr)-1)], name, color, ok and k>=len(fr)-3))
            row.append(np.full((row[-1].shape[0],4,3),255,np.uint8))
        frames.append(np.concatenate(row[:-1],axis=1))
    base=Path(a.out)/f"task{tid:02d}"
    imageio.mimwrite(str(base)+".mp4", frames, fps=20, codec="libx264", quality=8,
                     macro_block_size=None, ffmpeg_params=["-pix_fmt","yuv420p"])
    imageio.imwrite(str(base)+".png", frames[int(n*0.6)])
    print(f"[vid-{a.suite}] saved {base.name}.mp4 ({n}f)", flush=True)
print(f"[vid-{a.suite}] DONE ->", a.out, flush=True)
