#!/usr/bin/env bash
# Watch ens finish -> aggregate -> launch pin -> when POD ready launch train.
set -uo pipefail
BASE="/home/user/Desktop/Ayush PH test/DeepONet PH/v2"
CONTRIB="/home/user/Desktop/Ayush PH test/contrib_postjul15"
PY="/home/user/Desktop/Ayush PH test/venv/bin/python"
OUT="$BASE/exec_campaign_results"
LOG="$OUT/watcher.log"
exec >>"$LOG" 2>&1
echo "[$(date)] watcher start"

count_tasks() {
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  "$PY" - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
n=0
for b,bd in d.items():
  if b.startswith("_"): continue
  for m,md in bd.items():
    if isinstance(md,dict) and "per_task" in md:
      n=max(n,len(md["per_task"]))
print(n)
PY
}

while true; do
  n0=$(count_tasks "$OUT/m3_ens_t0-4/success_rates.json")
  n1=$(count_tasks "$OUT/m3_ens_t5-9/success_rates.json")
  echo "[$(date)] ens tasks: t0-4=$n0 t5-9=$n1"
  if [[ "$n0" -ge 5 && "$n1" -ge 5 ]]; then
    echo "ENS COMPLETE"
    break
  fi
  # also bail if processes died
  if ! pgrep -f 'exec ens' >/dev/null; then
    echo "ens processes died; aggregating whatever we have"
    break
  fi
  sleep 120
done

"$PY" - "$OUT" <<'PY'
import json, glob, os, sys
out=sys.argv[1]
modes={}
for f in sorted(glob.glob(os.path.join(out,"*","success_rates.json"))):
    name=os.path.basename(os.path.dirname(f))
    if name.startswith("_"): continue
    d=json.load(open(f))
    for bench,bd in d.items():
        if bench.startswith("_"): continue
        for m,md in bd.items():
            if not isinstance(md,dict) or "per_task" not in md: continue
            mode=md.get("exec_mode") or name.split("_")[1]
            modes.setdefault(mode,{}).update({k:v["success_rate"] for k,v in md["per_task"].items()})
summary={}
for mode,tasks in modes.items():
    rates=list(tasks.values())
    if not rates: continue
    summary[mode]={"mean_sr":round(100*sum(rates)/len(rates),2),"n_tasks":len(rates),
                   "per_task":{k:round(100*v,1) for k,v in sorted(tasks.items(), key=lambda x:int(x[0]))}}
summary["_anchors"]={"m3_r5":60.0,"flow_r5":65.0,"flow_reported":66.5}
json.dump(summary, open(os.path.join(out,"SUMMARY.json"),"w"), indent=2)
print(json.dumps(summary, indent=2))
PY

# Launch pin only (fixed action space)
export HF_HOME=/home/user/.cache/huggingface
export HF_TOKEN=$(tr -d '\r\n' < "$HF_HOME/token" || true)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export MUJOCO_GL=osmesa TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export DEEPONET_P=256 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=16 DEEPONET_HEAD=deeponet
export PYTHONPATH="$CONTRIB:$BASE"
M3="$BASE/deeponet_results/Long/runs/m3_s0/checkpoints/30000"
NORM=policy_preprocessor_step_5_normalizer_processor.safetensors
rm -rf "$OUT/m3_pin_t0-4" "$OUT/m3_pin_t5-9"
mkdir -p "$OUT/m3_pin_t0-4" "$OUT/m3_pin_t5-9"
run_pin(){
  local name=$1 tasks=$2
  OFFLINE_STATS_SF="$M3/$NORM" nohup "$PY" "$CONTRIB/eval_exec_offline.py" \
    --suite libero_10 --dataset lerobot/libero_10_image \
    --model "m3=deeponet=$M3" --exec pin --replan 5 \
    --indist_episodes 20 --task_ids "$tasks" --out "$OUT/$name" \
    >"$OUT/$name/run.log" 2>&1 &
  echo "started $name $!"
}
run_pin m3_pin_t0-4 0,1,2,3,4
run_pin m3_pin_t5-9 5,6,7,8,9

# Wait for POD asset then train
POD="$BASE/pod_assets/pod_p32.pt"
for i in $(seq 1 180); do
  [[ -f "$POD" ]] && break
  sleep 60
done
if [[ -f "$POD" ]]; then
  echo "POD ready — launching train"
  OUTT="$BASE/pod_train_spatial"
  mkdir -p "$OUTT"
  export DEEPONET_HEAD=pod POD_CKPT="$POD"
  nohup env PYTHONPATH="$CONTRIB:$BASE" DEEPONET_HEAD=pod POD_CKPT="$POD" \
    "$PY" "$BASE/train.py" \
    --head deeponet --variant baseline --deeponet_head pod \
    --deeponet_p 32 --deeponet_fourier 0 \
    --out "$OUTT" --dataset lerobot/libero_spatial_image --seed 0 \
    --stage1_steps 1650 --stage2_steps 6650 \
    >"$OUTT/train.log" 2>&1 &
  echo POD_TRAIN_PID=$!
else
  echo "POD not ready after timeout"
fi
echo "[$(date)] watcher done phase1"
touch "$OUT/WATCHER_PHASE1_DONE"
