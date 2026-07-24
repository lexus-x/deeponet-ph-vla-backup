#!/usr/bin/env bash
set -euo pipefail
echo "=== search init_files ==="
find /home/user -type d -name 'libero_spatial' 2>/dev/null | head -20
find /home/user -type d -name 'init_files' 2>/dev/null | head -20
echo "=== vla-atlas ==="
ls /home/user/vla-atlas/LIBERO/libero/libero/init_files 2>/dev/null | head || true
echo "=== Downloads ==="
ls /home/user/Downloads/libero 2>/dev/null | head || true
echo "=== conda saptarshi libero ==="
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate saptarshi
python - <<'PY'
import libero, os
print("libero file", libero.__file__)
root = os.path.dirname(libero.__file__)
print("root", root)
for p in [root, os.path.join(root, "libero"), os.path.join(root, "..")]:
    init = os.path.join(os.path.abspath(p), "init_files")
    print("check", init, "exists", os.path.isdir(init))
PY
echo "=== env vars ==="
env | grep -i libero || true
