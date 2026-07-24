#!/bin/bash
# Free CPU for POD: pin early results are catastrophic vs 60% anchor.
# Keep POD training; dump pin partial into SUMMARY.
set -u
ROOT="/home/user/Desktop/Ayush PH test/DeepONet PH/v2"
echo "Killing pin evals..."
pkill -f 'exec pin' || true
sleep 2
ps -ef | grep -E 'eval_exec|train\.py' | grep -v grep || true
echo "=== pin partial rates ==="
python3 - <<'PY'
import json, glob, os
root = "/home/user/Desktop/Ayush PH test/DeepONet PH/v2/exec_campaign_results"
summary_path = os.path.join(root, "SUMMARY.json")
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
pin = {}
for p in sorted(glob.glob(root + "/m3_pin_*/success_rates.json")):
    d = json.load(open(p))
    print(p, d)
    # expect dict task_id -> rate or nested
    if isinstance(d, dict):
        for k,v in d.items():
            if isinstance(v, (int,float)):
                pin[str(k)] = float(v)
            elif isinstance(v, dict) and "success_rate" in v:
                pin[str(k)] = float(v["success_rate"])
vals = list(pin.values())
summary["pin"] = {
    "mean_sr": (sum(vals)/len(vals) if vals else None),
    "n_tasks": len(vals),
    "per_task": pin,
    "status": "partial_aborted_for_pod_cpu",
    "note": "Early tasks catastrophic vs m3_r5=60; aborted to free osmesa/CPU for POD train",
}
json.dump(summary, open(summary_path,"w"), indent=2)
print("UPDATED", summary_path)
print(json.dumps(summary, indent=2))
PY
echo "=== POD log tail ==="
tail -20 "$ROOT/pod_train_spatial/train.log"
# ensure POD env
export MUJOCO_GL=osmesa
export POD_CKPT="$ROOT/pod_assets/pod_p32_a32.pt"
# if train died, restart
if ! pgrep -f 'pod_train_spatial' >/dev/null; then
  echo "POD dead — restarting"
  cd "$ROOT"
  nohup /home/user/Desktop/Ayush\ PH\ test/venv/bin/python train.py \
    --head deeponet --variant baseline --deeponet_head pod \
    --deeponet_p 32 --deeponet_fourier 0 \
    --out "$ROOT/pod_train_spatial" \
    --dataset lerobot/libero_spatial_image --seed 0 \
    --stage1_steps 1650 --stage2_steps 6650 \
    > "$ROOT/pod_train_spatial/train.log" 2>&1 &
  echo restarted pid $!
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
