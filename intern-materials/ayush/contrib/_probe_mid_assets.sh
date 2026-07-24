#!/usr/bin/env bash
set -u
echo "=== search libero/mujoco assets on mid ==="
find /mnt/d /home/islab -maxdepth 5 \( -iname 'libero' -type d -o -iname 'init_files' -type d -o -name 'libero_spatial' -type d \) 2>/dev/null | head -40
ls /mnt/d/Abhi /mnt/d/Abhineeth /mnt/d/WSL 2>/dev/null | head -40
# check if pip packages cached
ls ~/miniconda3/envs/lerobot/lib/python3.10/site-packages 2>/dev/null | grep -iE 'mujoco|robosuite|libero' || echo 'no packages'
