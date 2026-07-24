#!/usr/bin/env bash
# Fix LIBERO path expected by lerobot on a100, then re-run Spatial eval.
set -uo pipefail
# lerobot looked for /home/user/Desktop/LIBERO/libero/libero/init_files/...
SRC=/home/user/anaconda3/envs/saptarshi/lib/python3.12/site-packages/libero/libero
DST=/home/user/Desktop/LIBERO/libero/libero
mkdir -p "$(dirname "$DST")"
if [[ ! -e "$DST" ]]; then
  ln -sfn "$SRC" "$DST"
fi
ls -la "$DST/init_files/libero_spatial" | head
# also set config if present
mkdir -p "$HOME/.libero"
python - <<'PY'
import json, os
cfg = os.path.expanduser("~/.libero/config.yaml")
# libero often uses json-ish yaml; write simple paths
src = "/home/user/anaconda3/envs/saptarshi/lib/python3.12/site-packages/libero/libero"
open(cfg,"w").write(f"""datasets: {src}/../datasets
bddl_files: {src}/bddl_files
init_states: {src}/init_files
assets: {src}/assets
""")
print("wrote", cfg)
PY

DST_HOME=$HOME/deeponet_campaign
cd $DST_HOME/v2
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate saptarshi
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export DEEPONET_P=256 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=16 DEEPONET_HEAD=deeponet
export PYTHONPATH=$DST_HOME/contrib:$DST_HOME/v2:${PYTHONPATH:-}
M3=$DST_HOME/ckpts/m3_spatial_30k
FLOW=$DST_HOME/ckpts/flow_spatial_30k
OUT=$DST_HOME/plus_spatial_results
mkdir -p "$OUT"
NORM=policy_preprocessor_step_5_normalizer_processor.safetensors
echo START $(date) | tee "$OUT/run2.log"
OFFLINE_STATS_SF="$M3/$NORM" python "$DST_HOME/contrib/eval_exec_offline.py" \
  --suite libero_spatial --dataset lerobot/libero_spatial_image \
  --model "m3=deeponet=$M3" --exec none --replan 5 \
  --indist_episodes 10 --n_tasks 10 --out "$OUT/m3_indist" 2>&1 | tee -a "$OUT/run2.log"
OFFLINE_STATS_SF="$FLOW/$NORM" python "$DST_HOME/contrib/eval_exec_offline.py" \
  --suite libero_spatial --dataset lerobot/libero_spatial_image \
  --model "flow=flow=$FLOW" --exec none --replan 5 \
  --indist_episodes 10 --n_tasks 10 --out "$OUT/flow_indist" 2>&1 | tee -a "$OUT/run2.log"
echo ALL_DONE $(date) | tee -a "$OUT/run2.log"
