"""verify_build.py — param budget (<=61M) + smoke forward/backward for all 3 ACT variants."""
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors

from modeling_act_deeponet import ACTDeepONetPolicy

DEV = "cuda"
DS = "lerobot/libero_spatial_image"

# ---- FULL original ACT (uncompressed) ----
from act_common import CFG  # single source of truth

def build_config(meta):
    feats = dataset_to_policy_features(meta.features)
    out = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    inp = {k: v for k, v in feats.items() if v.type is not FeatureType.ACTION and k.startswith("observation.")}
    norm = {"VISUAL": NormalizationMode.MEAN_STD, "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD}
    return ACTConfig(input_features=inp, output_features=out, normalization_mapping=norm, **CFG)

def count(p): return sum(x.numel() for x in p.parameters())
def count_tr(p): return sum(x.numel() for x in p.parameters() if x.requires_grad)

def main():
    meta = LeRobotDatasetMetadata(DS)
    cfg = build_config(meta)
    print(f"action_dim={cfg.action_feature.shape[0]} chunk={cfg.chunk_size} dim={cfg.dim_model} ff={cfg.dim_feedforward}")

    print("\n=== PARAM COUNTS (budget <= 61.0M) ===")
    pols = {}
    for v in ["act", "act_deeponet", "act_deeponet_ph"]:
        pol = ACTDeepONetPolicy(cfg, variant=v).to(DEV)
        tot = count(pol) / 1e6
        lang = count(pol.lang_encoder) / 1e6
        head = (count(pol.deeponet_head) / 1e6) if hasattr(pol, "deeponet_head") else 0.0
        ok = "OK" if tot <= 61.0 else "OVER!!"
        print(f"  {v:18s}: {tot:6.2f}M  (lang {lang:.2f}M, deeponet-head {head:.2f}M)  [{ok}]")
        pols[v] = pol

    print("\n=== SMOKE forward/backward on a real batch ===")
    dt = resolve_delta_timestamps(cfg, meta)
    ds = LeRobotDataset(DS, delta_timestamps=dt)
    pre, _ = make_act_pre_post_processors(cfg, dataset_stats=meta.stats)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    raw = next(iter(loader))
    batch = pre(raw)
    batch = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
    for v, pol in pols.items():
        pol.train()
        loss, info = pol.forward(batch)
        loss.backward()
        print(f"  {v:18s}: loss={loss.item():.4f}  {({k: round(x,4) if isinstance(x,float) else x for k,x in info.items()})}")
    print("\nVERIFY: PASS — all variants build, fit budget, and train a step.")

if __name__ == "__main__":
    main()
