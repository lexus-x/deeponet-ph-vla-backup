#!/usr/bin/env python
"""Aggregate the Goal object-layout test: mean +/- std across layout slices
(overall + per-task). Each slice = a different set of object positions."""
import json, glob, numpy as np
from pathlib import Path

BASE = (Path(__file__).parent / ".." / "goal_layouttest").resolve()
files = sorted(glob.glob(str(BASE / "off_*" / "success_rates.json")))
avgs, per_task, labels = [], {}, []
print(f"[aggregate] {len(files)} layout-slice file(s) under {BASE}")
for f in files:
    d = json.load(open(f))
    g = d.get("LIBERO-GOAL", {}).get("flow", {})
    a = g.get("average")
    sl = Path(f).parent.name  # off_0 / off_20 / off_30
    ndone = len(g.get("per_task", {}))
    if a is None:
        print(f"  {sl}: incomplete ({ndone}/10 tasks)")
        continue
    avgs.append(a * 100); labels.append(sl)
    print(f"  {sl}: avg={a*100:.1f}%  ({ndone}/10 tasks)")
    for t, v in g.get("per_task", {}).items():
        per_task.setdefault(int(t), []).append(v["success_rate"] * 100)

if avgs:
    print(f"\n==== GOAL flow over {len(avgs)} object-layout slice(s) ====")
    print(f"SUITE AVERAGE: {np.mean(avgs):.1f}%  +/- {np.std(avgs):.1f}   slices={dict(zip(labels,[round(x,1) for x in avgs]))}")
    print("per-task (mean +/- std across layouts):")
    for t in sorted(per_task):
        v = per_task[t]
        print(f"  task{t}: {np.mean(v):5.1f}%  +/- {np.std(v):4.1f}   {[round(x) for x in v]}")
else:
    print("\n(no completed layout slices yet)")
