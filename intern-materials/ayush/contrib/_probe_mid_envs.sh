#!/usr/bin/env bash
set -u
source ~/miniconda3/etc/profile.d/conda.sh
echo "=== ENVS ==="
conda env list
for env in lerobot nora_lerobot; do
  echo "=== env=$env ==="
  conda activate "$env" || continue
  python - <<'PY'
import sys
print("py", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
except Exception as e:
    print("TORCH", e)
for mod in ("lerobot", "libero", "robosuite", "mujoco"):
    try:
        m = __import__(mod)
        print(mod, "OK")
    except Exception as e:
        print(mod, "FAIL", type(e).__name__, str(e)[:120])
PY
  conda deactivate
done
echo "=== LIBERO assets ==="
ls ~/nora/experiments/libero 2>/dev/null | head
find /home/islab -maxdepth 4 -type d -name 'init_files' 2>/dev/null | head
ls /home/islab/.libero 2>/dev/null
