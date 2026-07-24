#!/usr/bin/env bash
# PRIMARY restart bet: multi-suite LIBERO-Plus M3 vs flow on blackwell.
set -uo pipefail
ROOT="/home/user/Desktop/Ayush PH test"
V2="$ROOT/DeepONet PH/v2"
VENV="$ROOT/venv/bin/python"
OUT="$V2/plus_multisuite_campaign"
mkdir -p "$OUT"
cd "$V2"
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export DEEPONET_P=256 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=16 DEEPONET_HEAD=deeponet
export PYTHONPATH="$V2:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="$HOME/.libero_plus"

M3_SP="$V2/deeponet_results/Spatial/runs/m3_s0/checkpoints/30000"
FLOW_SP="$ROOT/DeepONet PH/paper_repro/Spatial/runs/flow_s0/checkpoints/30000"
M3_OBJ="$V2/deeponet_results/Object/runs/m3_s0/checkpoints/30000"
FLOW_OBJ="$ROOT/DeepONet PH/paper_repro/Object/runs/flow_s0/checkpoints/30000"
M3_LONG="$V2/deeponet_results/Long/runs/m3_s0/checkpoints/30000"
FLOW_LONG="$ROOT/DeepONet PH/paper_repro/Long/runs/flow_s0/checkpoints/30000"

# Fast signal: fewer cats / n_per_cat for smoke, then full. Use paper defaults.
N_PER_CAT=${N_PER_CAT:-12}
REPLAN=5

run_plus () {
  local tag=$1 head=$2 ckpt=$3 suite=$4
  local odir="$OUT/${tag}_${suite}"
  mkdir -p "$odir"
  if [[ -f "$odir/robustness_plus.json" ]]; then
    echo "SKIP $tag $suite (exists)"
    return 0
  fi
  if [[ ! -d "$ckpt" ]]; then
    echo "MISSING ckpt $ckpt — skip $tag $suite" | tee -a "$OUT/run.log"
    return 1
  fi
  echo "START $tag $suite $(date)" | tee -a "$OUT/run.log"
  "$VENV" evaluate_plus.py \
    --model "${tag}=${head}=${ckpt}" \
    --suite "$suite" --replan "$REPLAN" --n_per_cat "$N_PER_CAT" \
    --out "$odir" 2>&1 | tee -a "$odir/run.log" | tee -a "$OUT/run.log"
  echo "DONE $tag $suite $(date)" | tee -a "$OUT/run.log"
}

echo "CAMPAIGN_START $(date)" | tee "$OUT/run.log"

# Order: Object first (biggest paper hole + ckpts ready), then Long, then Spatial sanity
run_plus m3 deeponet "$M3_OBJ" libero_object
run_plus flow flow "$FLOW_OBJ" libero_object
run_plus m3 deeponet "$M3_LONG" libero_10
run_plus flow flow "$FLOW_LONG" libero_10
run_plus m3 deeponet "$M3_SP" libero_spatial
run_plus flow flow "$FLOW_SP" libero_spatial

"$VENV" - <<'PY'
import json, glob, os
root = "/home/user/Desktop/Ayush PH test/DeepONet PH/v2/plus_multisuite_campaign"
summary = {}
for p in sorted(glob.glob(root + "/*/robustness_plus.json")):
    d = json.load(open(p))
    tag = os.path.basename(os.path.dirname(p))
    # try common keys
    if isinstance(d, dict):
        summary[tag] = {k: d[k] for k in d if k in ("mean","average","overall","per_category","total") or True}
        # compact
        avg = d.get("average") or d.get("mean") or d.get("overall")
        if avg is None and "per_category" in d:
            vals = [v if isinstance(v,(int,float)) else v.get("success_rate", v.get("mean")) for v in d["per_category"].values()]
            vals = [float(x) for x in vals if x is not None]
            avg = sum(vals)/len(vals) if vals else None
        summary[tag] = {"path": p, "average": avg, "keys": list(d.keys())[:15]}
json.dump(summary, open(root + "/SUMMARY.json","w"), indent=2)
print(json.dumps(summary, indent=2)[:4000])
PY

echo "CAMPAIGN_DONE $(date)" | tee -a "$OUT/run.log"
