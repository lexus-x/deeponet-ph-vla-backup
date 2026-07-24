#!/usr/bin/env python
"""
train.py  (DeepONet campaign)
=============================
One trainer for the three head-to-head models, all under an IDENTICAL recipe so
the comparison is fair:

    M1  --head flow     --variant baseline               (flow-matching)
    M3  --head deeponet --variant baseline               (DeepONet)
    M4  --head deeponet --variant ph --lambda_ph 0.02    (DeepONet + PH)

Recipe (LIBERO-Spatial, batch 48, ~7.5 epochs = ~400k frames seen)
------------------------------------------------------------------
Stage 1 (head warm-up, backbone frozen) : ~1,650 steps, head lr 1e-4
Stage 2 (full-backbone fine-tune)       : ~6,650 steps, head lr 1e-4,
                                          backbone lr 1e-5 (500-step warmup),
                                          bf16 + gradient checkpointing
EMA(0.999) over all trainable (non-dead) params; checkpoints save EMA weights.

Logging mirrors the flow-matching runs (log_step.csv / log_epoch.csv) with the
same column names (flow_matching_loss column holds the MSE regression loss for
the DeepONet models) so the existing plotting utilities work for all three.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    """Seed python/numpy/torch for reproducible multi-seed comparison."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

DEV = "cuda"
REPO_DATA = "lerobot/libero_spatial_image"


# ----------------------------------------------------------------------------- policy
def build_policy(head, variant, base_ckpt, lambda_ph, ph_k, deeponet_p):
    ph_enabled = (variant == "ph")
    if head == "flow":
        from modeling_smolvla_ph import SmolVLAPHPolicy
        pol = SmolVLAPHPolicy.from_pretrained(
            base_ckpt, ph_enabled=ph_enabled, lambda_ph=lambda_ph, ph_k=ph_k)
    elif head == "deeponet":
        from modeling_smolvla_deeponet import SmolVLADeepONetPolicy
        pol = SmolVLADeepONetPolicy.from_pretrained(
            base_ckpt, ph_enabled=ph_enabled, lambda_ph=lambda_ph, ph_k=ph_k,
            deeponet_p=deeponet_p)
    else:
        raise ValueError(head)
    return pol.to(DEV)


def adapt_features(policy, meta):
    from modeling_smolvla_ph import adapt_policy_features_to_dataset
    return adapt_policy_features_to_dataset(policy, meta)


# ----------------------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average over all currently-and-future trainable params.

    Built over every param that is NOT permanently dead, so it is valid across
    the stage-1 -> stage-2 transition (frozen params simply equal their own EMA).
    """

    def __init__(self, policy, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.names = []
        is_dead = getattr(policy, "_is_dead", lambda n: False)
        for n, p in policy.named_parameters():
            if is_dead(n):
                continue
            self.shadow[n] = p.detach().float().clone()
            self.names.append(n)
        self._backup = None

    @torch.no_grad()
    def update(self, policy):
        d = self.decay
        for n, p in policy.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    @torch.no_grad()
    def store_and_copy(self, policy):
        """Back up live weights and load EMA weights (for eval/checkpoint)."""
        self._backup = {}
        for n, p in policy.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.copy_(self.shadow[n].to(p.dtype))

    @torch.no_grad()
    def restore(self, policy):
        if self._backup is None:
            return
        for n, p in policy.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = None

    def state(self):
        return {"decay": self.decay, "shadow": self.shadow}


# ----------------------------------------------------------------------------- helpers
def to_device(b, dev=DEV):
    if torch.is_tensor(b):
        return b.to(dev, non_blocking=True)
    if isinstance(b, dict):
        return {k: to_device(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(to_device(v, dev) for v in b)
    return b


def cyclic_loader(dataset, batch_size, num_workers):
    epoch = 0
    while True:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=True, persistent_workers=False)
        for batch in loader:
            yield batch, epoch
        epoch += 1


class CSVLogger:
    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(
                {k: row.get(k, "") for k in self.fields})


def vram_gb():
    return torch.cuda.max_memory_allocated() / 1e9


# ----------------------------------------------------------------------------- stage
def run_stage(stage_name, scfg, policy, ema, data_iter, preprocessor, out_dir,
              step_logger, epoch_logger, epoch_steps, ckpt_every,
              global_step_start, grad_clip=10.0):
    is_stage2 = "backbone_lr" in scfg
    head_lr = scfg["head_lr"]
    backbone_lr = scfg.get("backbone_lr", 0.0)
    warmup = scfg.get("warmup", 0)
    n_steps = scfg["steps"]
    if n_steps <= 0:
        print(f"=== {stage_name} skipped (0 steps) ===", flush=True)
        return global_step_start

    if is_stage2:
        policy.unfreeze_all()
        if scfg.get("grad_ckpt"):
            policy.enable_gradient_checkpointing()
    else:
        policy.freeze_backbone()

    groups = policy.param_groups(backbone_lr=backbone_lr, head_lr=head_lr)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-6)
    bb_idx = next((i for i, g in enumerate(optimizer.param_groups)
                   if g.get("name") == "backbone"), None)

    tc = policy.trainable_param_count()
    print(f"\n=== {stage_name} | steps={n_steps} batch={scfg['batch']} "
          f"trainable: head={tc['head']/1e6:.2f}M backbone={tc['backbone']/1e6:.1f}M ===",
          flush=True)

    policy.train()
    epoch_buf = []
    t_last = time.perf_counter()
    for s in range(n_steps):
        gstep = global_step_start + s
        if is_stage2 and bb_idx is not None:
            ramp = min(1.0, (s + 1) / max(1, warmup))
            optimizer.param_groups[bb_idx]["lr"] = backbone_lr * ramp

        batch, ep = next(data_iter)
        batch = to_device(preprocessor(batch))

        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ld = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            (p for p in policy.parameters() if p.requires_grad), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(policy)

        lr_head = next(g["lr"] for g in optimizer.param_groups if g["name"] == "head")
        lr_bb = optimizer.param_groups[bb_idx]["lr"] if bb_idx is not None else 0.0
        step_time = time.perf_counter() - t_last
        t_last = time.perf_counter()

        row = dict(global_step=gstep, stage=stage_name, dataset_epoch=ep,
                   flow_matching_loss=ld["flow_matching_loss"], l1_loss=ld["l1_loss"],
                   ph_loss=ld["ph_loss"], total_loss=ld["total_loss"],
                   lr_head=lr_head, lr_backbone=lr_bb, vram_gb=round(vram_gb(), 3),
                   grad_norm=round(float(gnorm), 4), step_time_s=round(step_time, 4))
        step_logger.log(row)
        epoch_buf.append(row)

        if (s + 1) % epoch_steps == 0:
            n = len(epoch_buf)
            agg = {k: sum(r[k] for r in epoch_buf) / n
                   for k in ("flow_matching_loss", "l1_loss", "ph_loss", "total_loss", "vram_gb")}
            epoch_logger.log(dict(epoch_global_step=gstep, stage=stage_name,
                                  logging_epoch=(gstep // epoch_steps), **agg))
            print(f"[{stage_name}] step {gstep:6d} | mse={agg['flow_matching_loss']:.4f} "
                  f"L1={agg['l1_loss']:.4f} PH={agg['ph_loss']:.4f} "
                  f"total={agg['total_loss']:.4f} | VRAM={agg['vram_gb']:.1f}GB "
                  f"| {step_time:.3f}s/it", flush=True)
            epoch_buf = []

        if (s + 1) % ckpt_every == 0 or (s + 1) == n_steps:
            save_checkpoint(policy, ema, preprocessor, out_dir, gstep + 1)

    return global_step_start + n_steps


def save_checkpoint(policy, ema, preprocessor, out_dir, gstep):
    cdir = Path(out_dir) / "checkpoints" / str(gstep)
    cdir.mkdir(parents=True, exist_ok=True)
    if ema is not None:
        ema.store_and_copy(policy)      # save EMA weights as the checkpoint
    policy.save_pretrained(cdir)
    if ema is not None:
        ema.restore(policy)
    try:
        preprocessor.save_pretrained(cdir)
    except Exception as e:
        print(f"[ckpt] processor save skipped ({e})", flush=True)
    (Path(out_dir) / "checkpoints" / "LATEST.txt").write_text(str(gstep))
    print(f"[ckpt] saved -> {cdir}  (EMA weights)", flush=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["flow", "deeponet"], required=True)
    ap.add_argument("--variant", choices=["baseline", "ph"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_ckpt", default="lerobot/smolvla_base")
    ap.add_argument("--dataset", default=REPO_DATA)
    ap.add_argument("--lambda_ph", type=float, default=0.02)
    ap.add_argument("--ph_k", type=int, default=8)
    ap.add_argument("--deeponet_p", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--stage1_steps", type=int, default=1650)
    ap.add_argument("--stage2_steps", type=int, default=6650)
    ap.add_argument("--stage1_batch", type=int, default=48)
    ap.add_argument("--stage2_batch", type=int, default=48)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--backbone_lr", type=float, default=1e-5)
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay (0 disables)")
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--epoch_steps", type=int, default=200)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] head={args.head} variant={args.variant} seed={args.seed} out={out_dir}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    policy = build_policy(args.head, args.variant, args.base_ckpt,
                          args.lambda_ph, args.ph_k, args.deeponet_p)

    meta = LeRobotDatasetMetadata(args.dataset)
    adapt_features(policy, meta)
    dt = resolve_delta_timestamps(policy.config, meta)
    dataset = LeRobotDataset(args.dataset, delta_timestamps=dt)
    norm_stats = torch.load(args.stats_path) if args.stats_path else meta.stats
    preprocessor, _post = make_smolvla_pre_post_processors(policy.config, dataset_stats=norm_stats)
    print(f"[train] dataset frames={dataset.num_frames} episodes={dataset.num_episodes} "
          f"-> 1 epoch = {dataset.num_frames/args.stage2_batch:.0f} steps @ batch {args.stage2_batch}",
          flush=True)
    total_frames = args.stage1_steps * args.stage1_batch + args.stage2_steps * args.stage2_batch
    print(f"[train] planned frames seen = {total_frames} "
          f"= {total_frames/dataset.num_frames:.2f} epochs", flush=True)

    ema = EMA(policy, decay=args.ema) if args.ema and args.ema > 0 else None

    cfg = {
        "stage1": dict(steps=args.stage1_steps, batch=args.stage1_batch, head_lr=args.head_lr),
        "stage2": dict(steps=args.stage2_steps, batch=args.stage2_batch, head_lr=args.head_lr,
                       backbone_lr=args.backbone_lr, warmup=args.warmup, grad_ckpt=True),
    }

    step_fields = ["global_step", "stage", "dataset_epoch", "flow_matching_loss", "l1_loss",
                   "ph_loss", "total_loss", "lr_head", "lr_backbone", "vram_gb",
                   "grad_norm", "step_time_s"]
    epoch_fields = ["epoch_global_step", "stage", "logging_epoch", "flow_matching_loss",
                    "l1_loss", "ph_loss", "total_loss", "vram_gb"]
    step_logger = CSVLogger(out_dir / "log_step.csv", step_fields)
    epoch_logger = CSVLogger(out_dir / "log_epoch.csv", epoch_fields)

    (out_dir / "run_config.json").write_text(json.dumps(
        {"args": vars(args), "stage_configs": cfg,
         "dataset_frames": dataset.num_frames,
         "planned_epochs": total_frames / dataset.num_frames}, indent=2, default=str))

    t0 = time.perf_counter()
    gstep = 0
    di = cyclic_loader(dataset, cfg["stage1"]["batch"], args.num_workers)
    gstep = run_stage("stage1", cfg["stage1"], policy, ema, di, preprocessor, out_dir,
                      step_logger, epoch_logger, args.epoch_steps, args.ckpt_every, gstep)
    di = cyclic_loader(dataset, cfg["stage2"]["batch"], args.num_workers)
    gstep = run_stage("stage2", cfg["stage2"], policy, ema, di, preprocessor, out_dir,
                      step_logger, epoch_logger, args.epoch_steps, args.ckpt_every, gstep)

    dt_min = (time.perf_counter() - t0) / 60
    print(f"\n[train] DONE head={args.head} variant={args.variant} steps={gstep} "
          f"wall={dt_min:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
