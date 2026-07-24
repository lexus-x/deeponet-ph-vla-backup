#!/usr/bin/env bash
set -u
echo "=== HOST ==="
uname -a
nvidia-smi -L
df -h / | tail -1
echo "=== HOME ==="
ls -la ~ | head -30
echo "=== SEARCH ==="
find /home /mnt/d /mnt/c/Users -maxdepth 4 \( \
  -iname '*libero*' -o -iname '*lerobot*' -o -iname '*conda*' -o -iname '*miniconda*' \
  -o -iname '*deeponet*' -o -iname 'saptarshi*' \) 2>/dev/null | head -50
echo "=== PYTHON ==="
which python3 || true
python3 -c 'import sys; print(sys.version)' 2>/dev/null || true
ls /mnt/d 2>/dev/null | head -20
echo "=== DONE ==="
