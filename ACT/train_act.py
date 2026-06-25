#!/usr/bin/env python
"""train_act.py — train ACT / ACT+DeepONet / ACT+DeepONet+PH on a LIBERO suite.
Single stage (ResNet backbone has ImageNet init): 30K steps, batch 64, EMA, ckpt every 5K."""
from __future__ import annotations
import argparse, csv, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.processor_act import make_act_pre_post_processors

from act_common import build_policy, build_config, save_ckpt
from ram_cache import maybe_ram_cache

DEV = "cuda"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class EMA:
    def __init__(self, policy, decay=0.999):
        self.decay = decay; self.shadow = {}; self._backup = None
        for n, p in policy.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, policy):
        for n, p in policy.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def store_and_copy(self, policy):
        self._backup = {}
        for n, p in policy.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone(); p.copy_(self.shadow[n].to(p.dtype))

    @torch.no_grad()
    def restore(self, policy):
        if self._backup is None: return
        for n, p in policy.named_parameters():
            if n in self._backup: p.copy_(self._backup[n])
        self._backup = None


def to_dev(b):
    return {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def cyclic(ds, bs, nw):
    # one persistent loader (don't respawn workers each epoch); deep prefetch to keep the GPU fed
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
                        drop_last=True, persistent_workers=(nw > 0),
                        prefetch_factor=(6 if nw > 0 else None))
    ep = 0
    while True:
        for batch in loader:
            yield batch, ep
        ep += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["act", "act_deeponet", "act_deeponet_ph"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--epoch_steps", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lambda_ph", type=float, default=0.02)
    ap.add_argument("--ph_k", type=int, default=8)
    ap.add_argument("--ph_warmup", type=int, default=5000)
    ap.add_argument("--ph_trigger", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true", help="on-the-fly image augmentation (training only)")
    ap.add_argument("--aug_mode", default="strong", choices=["strong", "mild"],
                    help="strong=photometric+crop+blur (original); mild=colour-only, no crop (geometry-preserving)")
    a = ap.parse_args()
    # The image-parquet dataloader (CPU decode) is the throughput bottleneck, not the GPU.
    # This box has 56 cores but the campaign requests 32, leaving ~20 idle while the GPU
    # starves at ~0% util. Scale workers up to feed it, reserving ~8 cores for the main
    # process (pre-proc / H2D transfer / EMA). Worker count does NOT change which samples
    # land in which batch (sampler is seeded), so cross-variant comparisons stay valid.
    if a.num_workers > 0:
        a.num_workers = min(max(a.num_workers, (os.cpu_count() or 16) - 8), 52)
        print(f"[train] num_workers -> {a.num_workers} (cores={os.cpu_count()})", flush=True)
    set_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[train] variant={a.variant} dataset={a.dataset} -> {out}", flush=True)

    meta = LeRobotDatasetMetadata(a.dataset)
    ph_kw = dict(lambda_ph=a.lambda_ph, ph_k=a.ph_k, ph_warmup=a.ph_warmup, ph_trigger=a.ph_trigger)
    policy = build_policy(meta, a.variant, **ph_kw).to(DEV)
    cfg = policy.config
    dt = resolve_delta_timestamps(cfg, meta)
    ds = LeRobotDataset(a.dataset, delta_timestamps=dt)
    # Decode each unique frame once into RAM (deterministic dataset, no augmentation),
    # so the ~36 re-decodes/run vanish and the GPU stops starving. Byte-identical data;
    # falls back to on-the-fly decode if the suite won't fit in memory.
    ds, serve_workers = maybe_ram_cache(ds, a.num_workers, log=lambda m: print(m, flush=True))
    if a.augment:
        from augment import AugmentWrapper
        img_keys = list(getattr(cfg, "image_features", None) or
                        ["observation.images.image", "observation.images.wrist_image"])
        ds = AugmentWrapper(ds, img_keys, mode=a.aug_mode)
        serve_workers = min(a.num_workers, 24)   # augmentation is CPU-heavy -> more serve workers
        print(f"[train] image augmentation ON (mode={a.aug_mode}) for {img_keys} (serve_workers={serve_workers})", flush=True)
    pre, _ = make_act_pre_post_processors(cfg, dataset_stats=meta.stats)
    tot = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"[train] params={tot:.2f}M frames={ds.num_frames} batch={a.batch} steps={a.steps}", flush=True)

    bb = [p for n, p in policy.named_parameters() if n.startswith("model.backbone") and p.requires_grad]
    rest = [p for n, p in policy.named_parameters() if not n.startswith("model.backbone") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": rest, "lr": a.lr, "name": "main"},
                             {"params": bb, "lr": a.lr_backbone, "name": "backbone"}],
                            betas=(0.9, 0.95), weight_decay=1e-4)
    ema = EMA(policy, a.ema) if a.ema > 0 else None

    fields = ["step", "epoch", "l1_loss", "kld_loss", "total_loss", "ph_active_frac", "ph_gate", "lr", "vram_gb", "step_s"]
    with open(out / "log_step.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    (out / "run_config.json").write_text(json.dumps({"args": vars(a), "params_M": tot}, indent=2, default=str))

    di = cyclic(ds, a.batch, serve_workers)
    policy.train(); t0 = time.perf_counter(); tlast = t0; buf = []
    for s in range(a.steps):
        ramp = min(1.0, (s + 1) / max(1, a.warmup))
        for g in opt.param_groups:
            g["lr"] = (a.lr if g["name"] == "main" else a.lr_backbone) * ramp
        batch, ep = next(di)
        batch = to_dev(pre(batch))
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, info = policy.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 10.0)
        opt.step()
        if ema is not None: ema.update(policy)
        now = time.perf_counter(); step_s = now - tlast; tlast = now
        row = dict(step=s + 1, epoch=ep, l1_loss=round(info.get("l1_loss", 0), 4),
                   kld_loss=round(info.get("kld_loss", 0), 4), total_loss=round(float(loss.item()), 4),
                   ph_active_frac=round(info.get("ph_active_frac", 0), 3), ph_gate=info.get("ph_gate", "-"),
                   lr=opt.param_groups[0]["lr"], vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   step_s=round(step_s, 4))
        buf.append(row)
        if (s + 1) % a.epoch_steps == 0:
            with open(out / "log_step.csv", "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerows(buf)
            avg_l1 = sum(r["l1_loss"] for r in buf) / len(buf)
            print(f"[{a.variant}] step {s+1:6d} | l1={avg_l1:.4f} total={row['total_loss']:.3f} "
                  f"phgate={row['ph_gate']} act={row['ph_active_frac']} | VRAM={row['vram_gb']}GB | {step_s:.3f}s/it", flush=True)
            buf = []
        if (s + 1) % a.ckpt_every == 0 or (s + 1) == a.steps:
            save_ckpt(policy, ema, out, s + 1, a.variant, ph_kw)
    print(f"[train] DONE {a.variant} steps={a.steps} wall={(time.perf_counter()-t0)/60:.1f}min -> {out}", flush=True)


if __name__ == "__main__":
    main()
