#!/usr/bin/env python
"""train_act_v2.py — V2 transfer protocol for the ACT campaign.

Two stages (driven by run_act_v2.sh):
  1) PRETRAIN on the 40-task union of all 4 LIBERO suites (--datasets a,b,c,d).
     Normalisation stats = aggregate of the 4 suites' stats (correct for a mixed corpus).
  2) FINE-TUNE on one suite (--dataset <suite> --init_from <pretrain_ckpt>); the suite's
     own stats are used (standard per-dataset normalisation), and the model re-adapts.

Reuses the V1 training loop/helpers from train_act.py (identical optimiser/EMA/logging),
so cross-version comparisons stay apples-to-apples. Single GPU, bf16, EMA.
"""
from __future__ import annotations
import argparse, csv, json, os, time
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from safetensors.torch import load_file

from act_common import build_policy, save_ckpt
from ram_cache import maybe_ram_cache
from train_act import EMA, set_seed, to_dev, cyclic, DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["act", "act_deeponet", "act_deeponet_ph"])
    ap.add_argument("--datasets", default=None, help="comma-sep repo_ids -> multi-task PRETRAIN")
    ap.add_argument("--dataset", default=None, help="single repo_id -> single-suite FINETUNE")
    ap.add_argument("--init_from", default=None, help="checkpoint dir to load weights from (finetune)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=15000)
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
    a = ap.parse_args()

    assert a.datasets or a.dataset, "give --datasets (pretrain) or --dataset (finetune)"
    dsets = [d.strip() for d in a.datasets.split(",")] if a.datasets else [a.dataset]
    multi = len(dsets) > 1

    if a.num_workers > 0:
        a.num_workers = min(max(a.num_workers, (os.cpu_count() or 16) - 8), 52)
        print(f"[v2] num_workers -> {a.num_workers} (cores={os.cpu_count()})", flush=True)
    set_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stage = "PRETRAIN(40-task)" if multi else f"FINETUNE({dsets[0]})"
    print(f"[v2] {stage} variant={a.variant} -> {out}", flush=True)

    # policy config from the first dataset (all 4 suites share identical features)
    meta0 = LeRobotDatasetMetadata(dsets[0])
    ph_kw = dict(lambda_ph=a.lambda_ph, ph_k=a.ph_k, ph_warmup=a.ph_warmup, ph_trigger=a.ph_trigger,
                 decouple_gripper=False)
    policy = build_policy(meta0, a.variant, **ph_kw).to(DEV)
    cfg = policy.config

    # finetune: load pretrained weights (same architecture -> strict load)
    if a.init_from:
        ck = Path(a.init_from)
        if ck.name in ("LATEST", "BEST") and (ck.parent / "LATEST.txt").exists():
            ck = ck.parent / (ck.parent / "LATEST.txt").read_text().strip()   # .../checkpoints/LATEST
        elif (ck / "LATEST.txt").exists() and not (ck / "model.safetensors").exists():
            ck = ck / (ck / "LATEST.txt").read_text().strip()                  # .../checkpoints
        policy.load_state_dict(load_file(str(ck / "model.safetensors")))
        print(f"[v2] loaded init weights <- {ck}", flush=True)

    # normalisation stats: aggregate across suites for the mixed pretrain corpus,
    # else the single suite's own stats.
    if multi:
        metas = [LeRobotDatasetMetadata(d) for d in dsets]
        stats = aggregate_stats([m.stats for m in metas])
        print(f"[v2] aggregated stats over {len(metas)} suites", flush=True)
    else:
        stats = meta0.stats
    torch.save(stats, out / "stats.pt")

    dt = resolve_delta_timestamps(cfg, meta0)
    if multi:
        ds = ConcatDataset([LeRobotDataset(d, delta_timestamps=dt) for d in dsets])
        serve_workers = a.num_workers          # 40-task corpus won't fit RAM cache -> stream
        n_frames = len(ds)
    else:
        ds = LeRobotDataset(dsets[0], delta_timestamps=dt)
        # RAM-cache the suite if it fits (250GB box): fine-tune re-reads the data ~10x,
        # so caching kills the video-decode bottleneck. maybe_ram_cache auto-falls back to
        # streaming for suites too big to fit (e.g. LIBERO-Long ~160GB). Byte-identical data.
        ds, serve_workers = maybe_ram_cache(ds, a.num_workers, log=lambda m: print(m, flush=True))
        n_frames = getattr(ds, "num_frames", len(ds))

    pre, _ = make_act_pre_post_processors(cfg, dataset_stats=stats)
    tot = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"[v2] params={tot:.2f}M frames={n_frames} batch={a.batch} steps={a.steps} "
          f"datasets={dsets}", flush=True)

    bb = [p for n, p in policy.named_parameters() if n.startswith("model.backbone") and p.requires_grad]
    rest = [p for n, p in policy.named_parameters() if not n.startswith("model.backbone") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": rest, "lr": a.lr, "name": "main"},
                             {"params": bb, "lr": a.lr_backbone, "name": "backbone"}],
                            betas=(0.9, 0.95), weight_decay=1e-4)
    ema = EMA(policy, a.ema) if a.ema > 0 else None

    fields = ["step", "epoch", "l1_loss", "kld_loss", "total_loss", "ph_active_frac", "ph_gate", "lr", "vram_gb", "step_s"]
    with open(out / "log_step.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    (out / "run_config.json").write_text(json.dumps(
        {"args": vars(a), "params_M": tot, "stage": stage, "datasets": dsets}, indent=2, default=str))

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
            print(f"[{a.variant}|{'pre' if multi else 'ft'}] step {s+1:6d} | l1={avg_l1:.4f} "
                  f"total={row['total_loss']:.3f} phgate={row['ph_gate']} | VRAM={row['vram_gb']}GB | {step_s:.3f}s/it", flush=True)
            buf = []
        if (s + 1) % a.ckpt_every == 0 or (s + 1) == a.steps:
            save_ckpt(policy, ema, out, s + 1, a.variant, ph_kw)
    print(f"[v2] DONE {stage} {a.variant} steps={a.steps} wall={(time.perf_counter()-t0)/60:.1f}min -> {out}", flush=True)


if __name__ == "__main__":
    main()
