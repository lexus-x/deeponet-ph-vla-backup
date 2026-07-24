#!/usr/bin/env python
"""
train.py
========
Two-stage fine-tuning of SmolVLA (+ optional Persistent-Homology regularizer)
on LIBERO-10 (lerobot/libero_10_image), on a Blackwell GPU.

Recipe (full mode), matching the project spec
----------------------------------------------
Stage 1 (head warm-up, backbone frozen):
    5,000 steps | batch 8 | head lr 1e-4 | bf16
Stage 2 (full-backbone fine-tune):
    30,000 steps | batch 4 | grad checkpointing | bf16
    backbone lr 1e-5 (linear 0->1e-5 warmup over first 500 steps)
    head     lr 1e-4
    AdamW with parameter groups for differential LRs

Smoke mode: Stage 1 = 500 steps, Stage 2 = 2,000 steps (2,500 total).

Two variants (select with --variant):
    baseline : vanilla flow-matching loss
    ph       : flow-matching + lambda_ph * PH_loss   (lambda_ph default 0.1)

Logging (CSV, the source of truth for plots.py):
    <out>/log_step.csv   per-step: flow/L1/PH/total loss, LRs, VRAM, grad-norm
    <out>/log_epoch.csv  per "logging epoch" (a fixed step-window; a true dataset
                         epoch is ~num_frames/bs ~ 25k steps, too coarse to plot)
Checkpoints: every 5,000 steps (full) / 500 (smoke) and at the end of each stage,
under <out>/checkpoints/<global_step>/  (also saves processors for eval).

Usage
-----
    python train.py --variant ph       --mode smoke
    python train.py --variant baseline --mode full
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from modeling_smolvla_ph import SmolVLAPHPolicy, adapt_policy_features_to_dataset

REPO_DATA = "lerobot/libero_10_image"
DEV = "cuda"


# ----------------------------------------------------------------------------- config
def stage_configs(mode: str):
    if mode == "full":
        return {
            "stage1": dict(steps=5_000, batch=8, head_lr=1e-4),
            "stage2": dict(steps=30_000, batch=4, head_lr=1e-4, backbone_lr=1e-5,
                           warmup=500, grad_ckpt=True),
            "ckpt_every": 5_000,
            "epoch_steps": 500,   # "logging epoch" window
        }
    elif mode == "smoke":
        return {
            "stage1": dict(steps=500, batch=8, head_lr=1e-4),
            "stage2": dict(steps=2_000, batch=4, head_lr=1e-4, backbone_lr=1e-5,
                           warmup=100, grad_ckpt=True),
            "ckpt_every": 500,
            "epoch_steps": 100,
        }
    raise ValueError(mode)


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
    """Infinite iterator over the dataset; yields (batch, epoch_index)."""
    epoch = 0
    while True:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=True, persistent_workers=False)
        for batch in loader:
            yield batch, epoch
        epoch += 1


class CSVLogger:
    def __init__(self, path: Path, fields):
        self.path = path
        self.fields = fields
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(
                {k: row.get(k, "") for k in self.fields})


def vram_gb():
    return torch.cuda.max_memory_allocated() / 1e9


# ----------------------------------------------------------------------------- train one stage
def run_stage(stage_name, scfg, policy, data_iter, preprocessor, out_dir,
              step_logger, epoch_logger, epoch_steps, ckpt_every,
              global_step_start, grad_clip=10.0, wandb_run=None):
    is_stage2 = "backbone_lr" in scfg
    head_lr = scfg["head_lr"]
    backbone_lr = scfg.get("backbone_lr", 0.0)
    warmup = scfg.get("warmup", 0)
    n_steps = scfg["steps"]
    if n_steps <= 0:
        print(f"=== {stage_name} skipped (0 steps) ===", flush=True)
        return global_step_start

    # --- trainable set + optimizer (differential LRs via param groups) -------
    if is_stage2:
        policy.unfreeze_all()
        if scfg.get("grad_ckpt"):
            policy.enable_gradient_checkpointing()
    else:
        policy.freeze_backbone()

    groups = policy.param_groups(backbone_lr=backbone_lr, head_lr=head_lr)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-6)
    # locate backbone group for warmup ramping
    bb_idx = next((i for i, g in enumerate(optimizer.param_groups)
                   if g.get("name") == "backbone"), None)

    tc = policy.trainable_param_count()
    print(f"\n=== {stage_name} | steps={n_steps} batch={scfg['batch']} "
          f"trainable: head={tc['head']/1e6:.1f}M backbone={tc['backbone']/1e6:.1f}M ===",
          flush=True)

    policy.train()
    epoch_buf = []  # accumulates per-step metric dicts for the current logging-epoch
    t_last = time.perf_counter()

    for s in range(n_steps):
        gstep = global_step_start + s

        # backbone LR linear warmup 0 -> backbone_lr over `warmup` steps
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

        lr_head = next(g["lr"] for g in optimizer.param_groups if g["name"] == "head")
        lr_bb = optimizer.param_groups[bb_idx]["lr"] if bb_idx is not None else 0.0
        step_time = time.perf_counter() - t_last
        t_last = time.perf_counter()

        row = dict(global_step=gstep, stage=stage_name, dataset_epoch=ep,
                   flow_matching_loss=ld["flow_matching_loss"],
                   l1_loss=ld["l1_loss"], ph_loss=ld["ph_loss"],
                   total_loss=ld["total_loss"], lr_head=lr_head, lr_backbone=lr_bb,
                   vram_gb=round(vram_gb(), 3), grad_norm=round(float(gnorm), 4),
                   step_time_s=round(step_time, 4))
        step_logger.log(row)
        epoch_buf.append(row)

        if wandb_run is not None:
            wandb_run.log({f"step/{k}": v for k, v in row.items()
                           if isinstance(v, (int, float))}, step=gstep)

        # ---- logging-epoch aggregation ----
        if (s + 1) % epoch_steps == 0:
            n = len(epoch_buf)
            agg = {k: sum(r[k] for r in epoch_buf) / n
                   for k in ("flow_matching_loss", "l1_loss", "ph_loss",
                             "total_loss", "vram_gb")}
            erow = dict(epoch_global_step=gstep, stage=stage_name,
                        logging_epoch=(gstep // epoch_steps), **agg)
            epoch_logger.log(erow)
            print(f"[{stage_name}] step {gstep:6d} | flow={agg['flow_matching_loss']:.4f} "
                  f"L1={agg['l1_loss']:.4f} PH={agg['ph_loss']:.4f} "
                  f"total={agg['total_loss']:.4f} | VRAM={agg['vram_gb']:.1f}GB "
                  f"| {step_time:.3f}s/it", flush=True)
            epoch_buf = []

        # ---- checkpoint ----
        if (s + 1) % ckpt_every == 0 or (s + 1) == n_steps:
            save_checkpoint(policy, preprocessor, out_dir, gstep + 1)

    return global_step_start + n_steps


def save_checkpoint(policy, preprocessor, out_dir, gstep):
    cdir = Path(out_dir) / "checkpoints" / str(gstep)
    cdir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(cdir)
    # Save processor too (eval can also rebuild from dataset stats, but this is exact).
    try:
        preprocessor.save_pretrained(cdir)
    except Exception as e:
        print(f"[ckpt] processor save skipped ({e})", flush=True)
    # maintain a 'latest' pointer
    (Path(out_dir) / "checkpoints" / "LATEST.txt").write_text(str(gstep))
    print(f"[ckpt] saved -> {cdir}", flush=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "ph"], required=True)
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base_ckpt", default="lerobot/smolvla_base")
    ap.add_argument("--lambda_ph", type=float, default=0.1)
    ap.add_argument("--ph_k", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--stage1_steps", type=int, default=None, help="override stage1 steps")
    ap.add_argument("--stage2_steps", type=int, default=None, help="override stage2 steps")
    ap.add_argument("--dataset", default=REPO_DATA, help="LeRobot dataset repo id")
    ap.add_argument("--stage1_batch", type=int, default=None)
    ap.add_argument("--stage2_batch", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None, help="override stage2 backbone LR warmup steps")
    ap.add_argument("--stats_path", default=None, help="normalization stats .pt (e.g. a checkpoint's shipped stats) instead of dataset stats")
    ap.add_argument("--head_lr", type=float, default=None, help="override stage2 head LR")
    ap.add_argument("--backbone_lr", type=float, default=None, help="override stage2 backbone LR")
    args = ap.parse_args()

    out_dir = Path(args.out or f"outputs/{args.variant}_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] variant={args.variant} mode={args.mode} out={out_dir}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ---- policy ----
    ph_enabled = (args.variant == "ph")
    policy = SmolVLAPHPolicy.from_pretrained(
        args.base_ckpt, ph_enabled=ph_enabled, lambda_ph=args.lambda_ph, ph_k=args.ph_k
    ).to(DEV)

    # ---- data ----
    meta = LeRobotDatasetMetadata(args.dataset)
    # Re-point the pretrained policy's features at the dataset
    # (image/wrist_image/state/action) before building the processor.
    adapt_policy_features_to_dataset(policy, meta)
    dt = resolve_delta_timestamps(policy.config, meta)
    dataset = LeRobotDataset(args.dataset, delta_timestamps=dt)
    norm_stats = torch.load(args.stats_path) if args.stats_path else meta.stats
    preprocessor, _post = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=norm_stats)
    print(f"[train] dataset frames={dataset.num_frames} episodes={dataset.num_episodes}",
          flush=True)

    cfg = stage_configs(args.mode)
    if args.stage1_steps is not None:
        cfg["stage1"]["steps"] = args.stage1_steps
    if args.stage2_steps is not None:
        cfg["stage2"]["steps"] = args.stage2_steps
    if args.stage1_batch is not None:
        cfg["stage1"]["batch"] = args.stage1_batch
    if args.stage2_batch is not None:
        cfg["stage2"]["batch"] = args.stage2_batch
    if args.warmup is not None:
        cfg["stage2"]["warmup"] = args.warmup
    if args.head_lr is not None:
        cfg["stage1"]["head_lr"] = args.head_lr
        cfg["stage2"]["head_lr"] = args.head_lr
    if args.backbone_lr is not None:
        cfg["stage2"]["backbone_lr"] = args.backbone_lr

    # ---- loggers ----
    step_fields = ["global_step", "stage", "dataset_epoch", "flow_matching_loss",
                   "l1_loss", "ph_loss", "total_loss", "lr_head", "lr_backbone",
                   "vram_gb", "grad_norm", "step_time_s"]
    epoch_fields = ["epoch_global_step", "stage", "logging_epoch",
                    "flow_matching_loss", "l1_loss", "ph_loss", "total_loss", "vram_gb"]
    step_logger = CSVLogger(out_dir / "log_step.csv", step_fields)
    epoch_logger = CSVLogger(out_dir / "log_epoch.csv", epoch_fields)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="smolvla-ph", name=f"{args.variant}_{args.mode}",
                               config=vars(args))

    (out_dir / "run_config.json").write_text(json.dumps(
        {"args": vars(args), "stage_configs": cfg,
         "dataset": args.dataset, "base_ckpt": args.base_ckpt}, indent=2, default=str))

    t0 = time.perf_counter()
    gstep = 0
    # Stage 1 (batch 8)
    di = cyclic_loader(dataset, cfg["stage1"]["batch"], args.num_workers)
    gstep = run_stage("stage1", cfg["stage1"], policy, di, preprocessor, out_dir,
                      step_logger, epoch_logger, cfg["epoch_steps"], cfg["ckpt_every"],
                      gstep, wandb_run=wandb_run)
    # Stage 2 (batch 4) - new loader for the new batch size
    di = cyclic_loader(dataset, cfg["stage2"]["batch"], args.num_workers)
    gstep = run_stage("stage2", cfg["stage2"], policy, di, preprocessor, out_dir,
                      step_logger, epoch_logger, cfg["epoch_steps"], cfg["ckpt_every"],
                      gstep, wandb_run=wandb_run)

    dt_min = (time.perf_counter() - t0) / 60
    print(f"\n[train] DONE variant={args.variant} total_steps={gstep} "
          f"wall={dt_min:.1f} min  -> {out_dir}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
