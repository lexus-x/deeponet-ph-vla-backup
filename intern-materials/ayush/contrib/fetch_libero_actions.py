"""Download ONLY the action data (parquet, no videos) of the LIBERO-Spatial
LeRobot dataset and export sliding-window chunks to data/chunks_spatial.npy.

ponytail: tries the dataset name Ayush trained on, falls back to the v3 combo.
"""
import os, sys, glob
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)
T, STRIDE = 50, 10

from huggingface_hub import snapshot_download

last_err = None
for repo in ["lerobot/libero_spatial_image", "HuggingFaceVLA/libero",
             "physical-intelligence/libero"]:
    try:
        path = snapshot_download(repo_id=repo, repo_type="dataset",
                                 allow_patterns=["data/**", "meta/**"])
        print("downloaded:", repo, "->", path)
        break
    except Exception as e:  # gated/missing -> try next
        last_err = e
        print("failed:", repo, "->", type(e).__name__, str(e)[:120])
else:
    sys.exit(f"all dataset candidates failed: {last_err}")

import pandas as pd

files = sorted(glob.glob(os.path.join(path, "data", "**", "*.parquet"),
                         recursive=True))
print(f"{len(files)} parquet files")
chunks = []
for f in files:
    df = pd.read_parquet(f, columns=None)
    acol = "action" if "action" in df.columns else None
    if acol is None:
        continue
    acts = np.stack(df[acol].to_numpy())                    # (L, A)
    ep_col = "episode_index" if "episode_index" in df.columns else None
    if ep_col is not None:
        for _, g in df.groupby(ep_col):
            a = np.stack(g[acol].to_numpy())
            for s in range(0, len(a) - T + 1, STRIDE):
                chunks.append(a[s:s + T])
    else:
        for s in range(0, len(acts) - T + 1, STRIDE):
            chunks.append(acts[s:s + T])

chunks = np.asarray(chunks, dtype=np.float32)
np.save(os.path.join(OUT, "chunks_spatial.npy"), chunks)
print("saved chunks:", chunks.shape, "->", os.path.join(OUT, "chunks_spatial.npy"))
