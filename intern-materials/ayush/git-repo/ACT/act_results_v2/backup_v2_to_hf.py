"""Back up the ACT V2 transfer campaign outputs (checkpoints+logs+plots) to HF
before local pruning. Idempotent: re-running re-uploads only changed files.
Uploads runs/ (8K finetune) and runs_ft15k/ (15K re-finetune) under v2/."""
import sys
from huggingface_hub import HfApi
api = HfApi()
REPO = "AyushShah1107/act-deeponet-libero-checkpoints"
ROOT = "/media/user/C2FE578FFE577A9D/ACT_v2"

jobs = [
    (f"{ROOT}/runs",        "v2/runs"),
    (f"{ROOT}/runs_ft15k",  "v2/runs_ft15k"),
]
for local, dst in jobs:
    print(f">> uploading {local}  ->  {REPO}:{dst}", flush=True)
    api.upload_folder(
        repo_id=REPO, repo_type="model",
        folder_path=local, path_in_repo=dst,
        commit_message=f"Add {dst} (ACT V2 campaign delta backup before prune)",
    )
    print(f"   done: {dst}", flush=True)
print("ALL_V2_HF_UPLOADS_DONE", flush=True)
