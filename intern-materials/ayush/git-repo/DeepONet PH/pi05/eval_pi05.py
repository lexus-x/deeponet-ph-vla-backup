"""
eval_pi05.py  --  in-dist + LIBERO-Plus evaluation for the pi0.5 SOTA-comparison
variants. Loads a pi0.5 policy (flow / DeepONet / DeepONet+PH) and reuses the
policy-agnostic v2 harnesses (evaluate.eval_indist + evaluate_plus.run_plus_for_policy).

IMPORTANT — import isolation:
  The in-dist harness uses the ORIGINAL LIBERO; the Plus harness uses the LIBERO-Plus
  drop-in `libero` package (libero_plus_wrapper forces it onto sys.path). Both cannot
  live in one interpreter — whichever imports `libero` first wins, and the other then
  reads the wrong init_files (the `..._table_1.pruned_init` FileNotFoundError bug).
  FIX: `--only both` re-invokes THIS script as two isolated subprocesses
  (`--only indist` and `--only plus`); each imports only its own harness, so `libero`
  resolves correctly in each. The parent merges the two partial results.

Contract: $RESULTS/<variant>__<suite>.json = {comparison, variant, suite, indist_avg, plus_avg}.
"""
from __future__ import annotations
import argparse, json, os, sys, subprocess
from pathlib import Path
import numpy as np
import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.types import NormalizationMode

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.abspath(os.path.join(_HERE, "..", "v2"))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
# NOTE: evaluate / evaluate_plus are imported LAZILY inside the single-path branches
# below so the two `libero` packages never collide in one interpreter.
from modeling_smolvla_ph import adapt_policy_features_to_dataset  # noqa: E402  (generic, no libero)
from modeling_pi05_deeponet import PI05DeepONetPolicy             # noqa: E402

DEV = "cuda"
N_TASKS = 10
DEFAULT_MAX_STEPS = 520
SUITE_MAP = {"Spatial": "libero_spatial", "Object": "libero_object",
             "Long": "libero_10", "Goal": "libero_goal"}
DATASET_MAP = {"Spatial": "lerobot/libero_spatial_image", "Object": "lerobot/libero_object_image",
               "Long": "lerobot/libero_10_image", "Goal": "lerobot/libero_goal_image"}


def _resolve_latest(ckpt_path: str) -> str:
    """libero-free mirror of evaluate.resolve_latest (LATEST/BEST pointer files)."""
    p = Path(ckpt_path)
    if p.name in ("LATEST", "BEST"):
        ptr = p.parent / f"{p.name}.txt"
        if not ptr.exists() and p.name == "BEST":
            ptr = p.parent / "LATEST.txt"
        if ptr.exists():
            return str(p.parent / ptr.read_text().strip())
    return str(p)


def load_policy_pi05(head, ckpt, dataset_name):
    ckpt = _resolve_latest(ckpt)
    if head == "flow":
        pol = PI05Policy.from_pretrained(ckpt)
    else:
        pol = PI05DeepONetPolicy.from_pretrained(ckpt, ph_enabled=False)
    meta = LeRobotDatasetMetadata(dataset_name)
    adapt_policy_features_to_dataset(pol, meta)
    for k in ("STATE", "ACTION"):
        if pol.config.normalization_mapping.get(k) == NormalizationMode.QUANTILES:
            pol.config.normalization_mapping[k] = NormalizationMode.MEAN_STD
    pol = pol.to(DEV).eval()
    pre, post = make_pi05_pre_post_processors(pol.config, dataset_stats=meta.stats)
    return pol, pre, post


def _partial_path(results, variant, suite, only):
    return Path(results) / f"_partial_{variant}__{suite}__{only}.json"


def run_single(args):
    """Run exactly ONE harness (indist or plus) in this (clean) interpreter."""
    suite = SUITE_MAP[args.suite]
    ds = DATASET_MAP[args.suite]
    scratch = Path(args.results) / f"_{args.variant}__{args.suite}_raw"
    scratch.mkdir(parents=True, exist_ok=True)
    policy, pre, post = load_policy_pi05(args.head, args.ckpt, ds)

    # Receding-horizon replan interval. MUST be set explicitly: pi0.5's config
    # default n_action_steps is 50 (nearly open-loop), whereas the SmolVLA study
    # evaluated at replan=5 (frequent closed-loop re-observation). Applying it here
    # to the shared policy object covers BOTH the in-dist and Plus paths (they reuse
    # this same policy). Held identical for flow / deeponet / deeponet_ph.
    policy.config.n_action_steps = args.replan
    print(f"[eval_pi05] {args.variant}/{args.suite} n_action_steps <- replan={args.replan}", flush=True)

    val = None
    if args.only == "indist":
        import evaluate as ev            # ORIGINAL LIBERO (no libero_plus_wrapper here)
        models = {args.variant: (policy, pre, post)}
        res = {}
        ev.eval_indist(models, args.indist_episodes, args.max_steps, res,
                       str(scratch / "indist.json"), suite, f"{suite}_indist", n_tasks=N_TASKS)
        val = res[f"{suite}_indist"][args.variant]["average"]
        key = "indist_avg"
    else:  # plus
        import evaluate_plus as evp       # LIBERO-Plus drop-in libero
        val = evp.run_plus_for_policy(policy, pre, post, suite,
                                      n_per_cat=args.plus_episodes,
                                      max_steps=args.max_steps, out_dir=str(scratch),
                                      replan=args.replan)
        key = "plus_avg"

    _partial_path(args.results, args.variant, args.suite, args.only).write_text(json.dumps({key: val}))
    print(f"[eval_pi05:{args.only}] {args.variant}/{args.suite} {key}={val}", flush=True)


def run_both(args):
    """Orchestrate two ISOLATED subprocesses, then merge. Imports no libero itself."""
    base = [sys.executable, os.path.abspath(__file__),
            "--variant", args.variant, "--head", args.head, "--ckpt", args.ckpt,
            "--suite", args.suite, "--results", args.results,
            "--indist_episodes", str(args.indist_episodes),
            "--plus_episodes", str(args.plus_episodes),
            "--max_steps", str(args.max_steps), "--replan", str(args.replan)]
    vals = {"indist_avg": None, "plus_avg": None}
    for only in ("indist", "plus"):
        rc = subprocess.run(base + ["--only", only]).returncode
        p = _partial_path(args.results, args.variant, args.suite, only)
        if rc == 0 and p.exists():
            vals.update(json.loads(p.read_text()))
            p.unlink()
        else:
            print(f"[eval_pi05] WARNING: {only} subprocess rc={rc} (result left null)", flush=True)

    out = {"comparison": "c1", "variant": args.variant, "suite": args.suite,
           "indist_avg": vals["indist_avg"], "plus_avg": vals["plus_avg"]}
    Path(args.results, f"{args.variant}__{args.suite}.json").write_text(json.dumps(out, indent=2))
    print(f"[eval_pi05] {args.variant}/{args.suite} indist={vals['indist_avg']} plus={vals['plus_avg']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--head", choices=["flow", "deeponet"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", required=True, choices=list(SUITE_MAP))
    ap.add_argument("--results", required=True)
    ap.add_argument("--indist_episodes", type=int, default=12)
    ap.add_argument("--plus_episodes", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--only", choices=["indist", "plus", "both"], default="both")
    args = ap.parse_args()
    Path(args.results).mkdir(parents=True, exist_ok=True)

    if args.only == "both":
        run_both(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
