"""Upload ONLY the ACT checkpoints not already on HF (aug runs + gripfix ablation)
to AyushShah1107/act-deeponet-libero-checkpoints under extra/.  Idempotent."""
from huggingface_hub import HfApi
api = HfApi()
REPO = "AyushShah1107/act-deeponet-libero-checkpoints"
ROOT = "/home/user/Desktop/Ayush PH test/ACT"

jobs = [
    # (local folder, path in repo)
    (f"{ROOT}/act_results_aug", "extra/act_results_aug"),
    (f"{ROOT}/act_results_gripfix/Object/runs/act_deeponet/checkpoints/30000",
     "extra/act_results_gripfix/Object_act_deeponet_30000"),
]
for local, dst in jobs:
    print(f">> uploading {local}  ->  {REPO}:{dst}", flush=True)
    api.upload_folder(
        repo_id=REPO, repo_type="model",
        folder_path=local, path_in_repo=dst,
        commit_message=f"Add {dst} (delta backup: not previously on HF)",
    )
    print(f"   done: {dst}", flush=True)
print("ALL_HF_UPLOADS_DONE", flush=True)
