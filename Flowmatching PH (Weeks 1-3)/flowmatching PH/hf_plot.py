#!/usr/bin/env python
"""hf_plot.py - per-task success plot + summary from a lerobot eval_info.json."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def task_success(metrics):
    for k in ("successes", "success", "is_success"):
        if k in metrics:
            v = metrics[k]
            return float(np.mean(v)) * 100 if isinstance(v, list) else float(v) * 100
    # fall back to max_rewards (1.0 == success in LIBERO)
    if "max_rewards" in metrics:
        return float(np.mean(metrics["max_rewards"])) * 100
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_info", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="lerobot/smolvla_libero")
    args = ap.parse_args()
    d = json.loads(Path(args.eval_info).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # group per_task by task_group
    groups = {}
    for e in d.get("per_task", []):
        groups.setdefault(e["task_group"], {})[e["task_id"]] = task_success(e["metrics"])

    summary = {"overall": d.get("overall", {}).get("pc_success"), "per_group": {}, "per_task": {}}
    for g, tasks in groups.items():
        ids = sorted(tasks)
        vals = [tasks[i] for i in ids]
        avg = float(np.mean(vals))
        summary["per_group"][g] = avg
        summary["per_task"][g] = {str(i): tasks[i] for i in ids}
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(ids, vals, color="#2ca02c")
        ax.axhline(avg, ls="--", color="k", alpha=0.6, label=f"avg {avg:.1f}%")
        ax.set_xticks(ids); ax.set_xticklabels([f"t{i}" for i in ids])
        ax.set_ylim(0, 105); ax.set_ylabel("success rate (%)")
        ax.set_title(f"{args.title} on {g}: per-task success (avg {avg:.1f}%)")
        ax.grid(alpha=0.3, axis="y"); ax.legend()
        for i, v in zip(ids, vals):
            ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"per_task_{g}.{ext}", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[hf-plot] {g}: avg {avg:.1f}%  -> per_task_{g}.png/.pdf")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[hf-plot] overall pc_success = {summary['overall']}  -> {out}/summary.json")


if __name__ == "__main__":
    main()
