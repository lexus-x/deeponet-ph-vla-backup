#!/bin/bash
# Sync POD mid-ckpt + POD code from blackwell paths to a100, patch modeling, run Spatial eval.
# Called ON a100 after files arrive, OR orchestrated from Windows with scp -3 first.
set -uo pipefail
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate saptarshi
DST=$HOME/deeponet_campaign
mkdir -p "$DST/ckpts/pod_spatial_mid" "$DST/pod_assets" "$DST/pod_eval_spatial_mid"

# Patch modeling to support head_type=pod using PYTHONPATH contrib
python - <<'PY'
from pathlib import Path
p = Path.home() / "deeponet_campaign/v2/modeling_smolvla_deeponet_v2.py"
text = p.read_text()
needle = 'if head_type == "reg":'
pod_block = '''if head_type == "pod":
            import os
            from pod_trunk import PODHead
            pod = torch.load(os.environ["POD_CKPT"], map_location="cpu", weights_only=False)
            # clone buffers so safetensors load does not share storage
            mean = pod["mean"].detach().clone()
            basis = pod["basis"].detach().clone()
            self.deeponet = PODHead(
                context_dim=context_dim, mean=mean, basis=basis,
                d_model=d_model, n_queries=n_queries, n_blocks=n_blocks)
        el''' + 'if head_type == "reg":'
if 'head_type == "pod"' in text:
    print("modeling already has pod")
else:
    if needle not in text:
        raise SystemExit("needle missing")
    p.write_text(text.replace(needle, pod_block, 1))
    print("patched modeling for pod")
PY

export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export DEEPONET_P=32 DEEPONET_BLOCKS=3 DEEPONET_QUERIES=8 DEEPONET_FOURIER=0
export DEEPONET_HEAD=pod
export POD_CKPT=$DST/pod_assets/pod_p32_a32.pt
export PYTHONPATH=$DST/contrib:$DST/v2:${PYTHONPATH:-}
CKPT=$DST/ckpts/pod_spatial_mid
OUT=$DST/pod_eval_spatial_mid
NORM=policy_preprocessor_step_5_normalizer_processor.safetensors
test -f "$CKPT/model.safetensors" || { echo missing ckpt; exit 1; }
test -f "$POD_CKPT" || { echo missing pod basis; exit 1; }

echo START $(date) | tee "$OUT/run.log"
OFFLINE_STATS_SF="$CKPT/$NORM" python "$DST/contrib/eval_exec_offline.py" \
  --suite libero_spatial --dataset lerobot/libero_spatial_image \
  --model "pod=deeponet=$CKPT" --exec none --replan 5 \
  --indist_episodes 10 --n_tasks 10 --out "$OUT" 2>&1 | tee -a "$OUT/run.log"
echo DONE $(date) | tee -a "$OUT/run.log"
