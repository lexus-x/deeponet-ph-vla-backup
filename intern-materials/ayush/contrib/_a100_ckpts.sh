#!/bin/bash
set -u
echo "=== ckpts ==="
ls -la ~/deeponet_campaign/ckpts/ 2>/dev/null
find ~/deeponet_campaign -maxdepth 3 -type d -name '30000' 2>/dev/null
find ~/deeponet_campaign -maxdepth 2 -type d 2>/dev/null
echo "=== libero plus on a100 ==="
ls "/home/user/Desktop/Ayush PH test/third_party/LIBERO-plus" 2>/dev/null | head || echo missing_desktop
find /home/user -maxdepth 4 -type d -name 'LIBERO-plus' 2>/dev/null | head
echo "=== conda saptarshi python ==="
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate saptarshi
which python
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
echo "=== lplus env ==="
conda activate lerobot_lplus
which python
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1 | head
