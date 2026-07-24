#!/bin/bash
set -u
echo "=== A100 GPU ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
echo "=== PROCESSES ==="
ps -ef | grep -E 'evaluate|train\.py|plus' | grep -v grep || true
echo "=== SPATIAL RESULTS ==="
ls -la ~/deeponet_campaign/plus_spatial_results/ 2>/dev/null
cat ~/deeponet_campaign/plus_spatial_results/SUMMARY_spatial_indist.json 2>/dev/null
echo "=== LIBERO-PLUS CHECK ==="
cd ~/deeponet_campaign/v2 || exit 1
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /home/user/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate saptarshi 2>/dev/null || true
python - <<'PY'
import sys
sys.path.insert(0, "/home/user/deeponet_campaign/v2")
try:
    import libero_plus_wrapper as LP
    print("libero_plus_wrapper OK")
    print("CATEGORIES", getattr(LP, "CATEGORIES", None))
except Exception as e:
    print("FAIL", type(e).__name__, e)
PY
ls /home/user/vla-atlas/LIBERO 2>/dev/null | head
find /home/user -maxdepth 5 -type d -iname '*plus*' 2>/dev/null | head -20
