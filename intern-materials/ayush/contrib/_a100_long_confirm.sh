#!/usr/bin/env bash
set -uo pipefail
DST=$HOME/deeponet_campaign
cd $DST/v2
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate saptarshi
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export DEEPONET_P=256 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=16 DEEPONET_HEAD=deeponet
export PYTHONPATH=$DST/contrib:$DST/v2:${PYTHONPATH:-}
M3=$DST/ckpts/m3_long_30k
FLOW=$DST/ckpts/flow_long_30k
OUT=$DST/long_confirm_results
mkdir -p "$OUT"
NORM=policy_preprocessor_step_5_normalizer_processor.safetensors
echo START $(date) | tee "$OUT/run.log"
OFFLINE_STATS_SF="$M3/$NORM" python "$DST/contrib/eval_exec_offline.py" \
  --suite libero_10 --dataset lerobot/libero_10_image \
  --model "m3=deeponet=$M3" --exec none --replan 5 \
  --indist_episodes 10 --n_tasks 10 --out "$OUT/m3_long" 2>&1 | tee -a "$OUT/run.log"
OFFLINE_STATS_SF="$FLOW/$NORM" python "$DST/contrib/eval_exec_offline.py" \
  --suite libero_10 --dataset lerobot/libero_10_image \
  --model "flow=flow=$FLOW" --exec none --replan 5 \
  --indist_episodes 10 --n_tasks 10 --out "$OUT/flow_long" 2>&1 | tee -a "$OUT/run.log"
python - <<'PY'
import json,glob,os
out=os.path.expanduser("~/deeponet_campaign/long_confirm_results")
summary={}
for name in ["m3_long","flow_long"]:
    p=os.path.join(out,name,"success_rates.json")
    if not os.path.exists(p):
        continue
    d=json.load(open(p))
    # navigate LIBERO-10 / m3 or flow
    for suite, models in d.items():
        if suite.startswith("_"): continue
        for mid, md in models.items():
            pt=md.get("per_task",{})
            rates={k:(v["success_rate"]*100 if isinstance(v,dict) and v.get("success_rate",1)<=1 else (v.get("success_rate") if isinstance(v,dict) else v)) for k,v in pt.items()}
            # normalize to percent
            rates2={}
            for k,v in rates.items():
                fv=float(v)
                rates2[k]=fv*100 if fv<=1.0 else fv
            vals=list(rates2.values())
            summary[name]={"mean_sr": sum(vals)/len(vals) if vals else None, "n":len(vals), "per_task":rates2}
json.dump(summary, open(os.path.join(out,"SUMMARY_long_confirm.json"),"w"), indent=2)
print(json.dumps(summary, indent=2))
PY
echo ALL_DONE $(date) | tee -a "$OUT/run.log"
