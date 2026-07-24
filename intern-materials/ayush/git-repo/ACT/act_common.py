"""act_common.py — shared config builder + checkpoint save/load for the ACT campaign."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig
from modeling_act_deeponet import ACTDeepONetPolicy

# FULL original ACT (uncompressed): ff=3200, 7 decoder layers, chunk=100.
# (budget cap removed per user request — performance over the 61M limit)
CFG = dict(dim_model=512, dim_feedforward=3200, n_encoder_layers=4, n_decoder_layers=7,
           n_vae_encoder_layers=4, chunk_size=100, n_action_steps=100, latent_dim=32,
           vision_backbone="resnet18", kl_weight=10.0)


def build_config(meta):
    feats = dataset_to_policy_features(meta.features)
    out = {k: v for k, v in feats.items() if v.type is FeatureType.ACTION}
    inp = {k: v for k, v in feats.items()
           if v.type is not FeatureType.ACTION and k.startswith("observation.")}
    norm = {"VISUAL": NormalizationMode.MEAN_STD, "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD}
    return ACTConfig(input_features=inp, output_features=out, normalization_mapping=norm, **CFG)


def build_policy(meta, variant, **ph_kw):
    cfg = build_config(meta)
    return ACTDeepONetPolicy(cfg, variant=variant, **ph_kw)


def save_ckpt(policy, ema, out_dir, step, variant, ph_kw):
    cdir = Path(out_dir) / "checkpoints" / str(step)
    cdir.mkdir(parents=True, exist_ok=True)
    if ema is not None:
        ema.store_and_copy(policy)
    sd = {k: v.detach().cpu().contiguous() for k, v in policy.state_dict().items()}
    save_file(sd, str(cdir / "model.safetensors"))
    if ema is not None:
        ema.restore(policy)
    (cdir / "meta.json").write_text(json.dumps({"variant": variant, "cfg": CFG, "ph_kw": ph_kw}, indent=2))
    (Path(out_dir) / "checkpoints" / "LATEST.txt").write_text(str(step))
    print(f"[ckpt] saved -> {cdir} (EMA weights)", flush=True)


def load_ckpt(meta, ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.name in ("LATEST", "BEST") and (ckpt_dir.parent / "LATEST.txt").exists():
        ckpt_dir = ckpt_dir.parent / (ckpt_dir.parent / "LATEST.txt").read_text().strip()  # .../checkpoints/LATEST
    elif (ckpt_dir / "LATEST.txt").exists() and not (ckpt_dir / "model.safetensors").exists():
        ckpt_dir = ckpt_dir / (ckpt_dir / "LATEST.txt").read_text().strip()                 # .../checkpoints
    m = json.loads((ckpt_dir / "meta.json").read_text())
    pol = build_policy(meta, m["variant"], **m.get("ph_kw", {}))
    pol.load_state_dict(load_file(str(ckpt_dir / "model.safetensors")))
    return pol, m["variant"]
