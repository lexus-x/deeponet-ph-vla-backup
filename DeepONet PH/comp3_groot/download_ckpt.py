import os
from huggingface_hub import snapshot_download
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
p = snapshot_download(
    repo_id="nvidia/GR00T-N1.6-3B",
    local_dir="/media/user/C2FE578FFE577A9D/hf_cache/GR00T-N1.6-3B",
    cache_dir="/media/user/C2FE578FFE577A9D/hf_cache",
    max_workers=2,   # gentle on disk while comp-1 reads datasets
)
print("DOWNLOADED ->", p)
