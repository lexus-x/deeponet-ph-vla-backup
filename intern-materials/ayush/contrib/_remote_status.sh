#!/bin/bash
set -u
ROOT="/home/user/Desktop/Ayush PH test/DeepONet PH/v2"
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo "=== PROCESSES ==="
ps -ef | grep -E 'train\.py|evaluate_exec|eval_exec' | grep -v grep || true
echo "=== POD TRAIN ==="
if [[ -f "$ROOT/pod_train_spatial/train.log" ]]; then
  wc -l "$ROOT/pod_train_spatial/train.log"
  tail -50 "$ROOT/pod_train_spatial/train.log"
  ls -la "$ROOT/pod_train_spatial/checkpoints" 2>/dev/null | head -20 || echo "(no ckpts yet)"
else
  echo "no train.log"
fi
echo "=== PIN PROGRESS ==="
for d in "$ROOT"/exec_campaign_results/m3_pin_*; do
  echo "== $d"
  ls "$d" 2>/dev/null
  grep -E 'LIBERO-10|mean|task' "$d"/run.log 2>/dev/null | tail -20
done
echo "=== ENS SUMMARY ==="
python3 - <<'PY'
import json, glob, os
root = "/home/user/Desktop/Ayush PH test/DeepONet PH/v2/exec_campaign_results"
for p in sorted(glob.glob(root + "/**/results.json", recursive=True)):
    try:
        d = json.load(open(p))
        print(p, "->", {k:d.get(k) for k in ("mean_success","success_rate","n_episodes","mode") if k in d or True})
        print("  keys", list(d.keys())[:20])
        if "per_task" in d:
            print("  per_task", d["per_task"])
        if "mean" in d:
            print("  mean", d["mean"])
    except Exception as e:
        print(p, e)
for p in sorted(glob.glob(root + "/**/SUMMARY*.json", recursive=True)):
    print("SUMMARY", p)
    print(open(p).read()[:2000])
PY
echo "=== DONE ==="
