#!/usr/bin/env bash
set -u
source ~/miniconda3/etc/profile.d/conda.sh
for env in nora nora_train onet base llava; do
  echo "=== env=$env ==="
  conda activate "$env" 2>/dev/null || { echo skip; continue; }
  python - <<'PY'
for mod in ("torch","libero","robosuite","mujoco","lerobot"):
    try:
        m=__import__(mod)
        extra=""
        if mod=="torch":
            import torch
            extra=f" cuda={torch.cuda.is_available()} n={torch.cuda.device_count()}"
        print(mod,"OK",extra)
    except Exception as e:
        print(mod,"FAIL",type(e).__name__)
PY
  conda deactivate
done
