#!/usr/bin/env python
"""
evaluate_exec.py
================
Closed-loop LIBERO eval with post-Jul-15 chunk-execution modes for DeepONet:

  none     — stock receding-horizon (Ayush baseline)
  pin      — boundary pinning a_new(0)=a_old(switch), cosine decay
  ens      — ACT-style exponential blend of overlapping chunks
  pinens   — pin then ensemble
  bid      — Operator-BID: K state-noise candidates, pick by backward coherence

No episode videos (speed). Resumable JSON. Task sharding via --task_ids.
Designed to sit next to evaluate.py / eval_offline.py on the GPU box.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "osmesa")

# Local imports from the v2 campaign dir (cwd or same folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402
from operator_bid import select_bid_chunk  # noqa: E402
from operator_chunk_exec import pin_boundary  # noqa: E402

DEV = "cuda"


def _cosine_ramp(tau: torch.Tensor, decay_tau: float) -> torch.Tensor:
    w = torch.clamp(tau / max(decay_tau, 1e-6), 0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * w))


class ExecController:
    """Wraps a loaded DeepONet/flow policy with optional pin/ensemble execution."""

    def __init__(self, policy, pre, post, mode: str, replan: int,
                 chunk_size: int = 50, decay_tau: float = 0.25, ens_alpha: float = 0.5,
                 ens_max: int = 4, bid_k: int = 8, bid_noise: float = 0.01):
        self.policy = policy
        self.pre = pre
        self.post = post
        self.mode = mode
        self.replan = int(replan)
        self.chunk_size = int(chunk_size)
        self.decay_tau = float(decay_tau)
        self.ens_alpha = float(ens_alpha)
        self.ens_max = int(ens_max)
        self.bid_k = int(bid_k)
        self.bid_noise = float(bid_noise)
        self.reset()

    def reset(self):
        self.policy.reset()
        self._queue: deque[np.ndarray] = deque()
        self._last_action: torch.Tensor | None = None
        self._chunks: list[tuple[torch.Tensor, int]] = []  # (chunk TxA cpu, start_step)
        self._step = 0

    def _predict_chunk(self, pin: dict) -> torch.Tensor:
        """Return (T, A) float32 CPU tensor of the predicted action chunk.

        Temporarily expand n_action_steps to the full chunk horizon so pinning /
        ensembling see the whole trajectory (stock select_action only queues `replan`).
        """
        cfg = self.policy.config
        old_n = getattr(cfg, "n_action_steps", self.replan)
        cfg.n_action_steps = max(self.chunk_size, old_n)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if hasattr(self.policy, "predict_action_chunk"):
                    chunk = self.policy.predict_action_chunk(pin)
                elif hasattr(self.policy, "_get_action_chunk"):
                    chunk = self.policy._get_action_chunk(pin)
                else:
                    raise RuntimeError("policy has no predict_action_chunk")
        finally:
            cfg.n_action_steps = old_n
        if isinstance(chunk, dict):
            chunk = chunk.get("action", chunk.get("actions"))
        chunk = chunk.detach().float().cpu()
        if chunk.ndim == 3:
            chunk = chunk[0]
        return chunk  # (T, A)

    def _apply_pin(self, chunk: torch.Tensor) -> torch.Tensor:
        if self._last_action is None or self.mode not in ("pin", "pinens"):
            return chunk
        T = chunk.shape[0]
        a_old = self._last_action.reshape(1, -1)  # (1, A)
        a_new_0 = chunk[0:1]
        offset0 = a_old - a_new_0
        taus = torch.linspace(0, 1, T)
        ramp = _cosine_ramp(taus, self.decay_tau).view(T, 1)
        return chunk + offset0 * ramp

    def _ensembled_action(self, step: int) -> torch.Tensor | None:
        preds, ws = [], []
        for ch, start in self._chunks:
            k = step - start
            if 0 <= k < ch.shape[0]:
                preds.append(ch[k])
                ws.append(math.exp(-self.ens_alpha * k))
        if not preds:
            return None
        w = torch.tensor(ws, dtype=torch.float32)
        w = w / w.sum()
        stacked = torch.stack(preds, 0)
        return (stacked * w.view(-1, 1)).sum(0)

    def act(self, obs, task_description) -> np.ndarray:
        if len(self._queue) == 0:
            pin = self.pre(ev.env_obs_to_policy_input(obs, task_description))
            pin = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
            if self.mode == "bid":
                raw = select_bid_chunk(
                    self._predict_chunk, pin, self._last_action,
                    k=self.bid_k, noise_std=self.bid_noise,
                )
                pinned = raw
            else:
                raw = self._predict_chunk(pin)
                pinned = self._apply_pin(raw)

            if self.mode in ("ens", "pinens"):
                self._chunks.append((pinned, self._step))
                self._chunks = self._chunks[-self.ens_max:]
                # fill queue with ensembled actions for the next replan window
                for i in range(self.replan):
                    a = self._ensembled_action(self._step + i)
                    if a is None:
                        a = pinned[min(i, pinned.shape[0] - 1)]
                    self._queue.append(a.numpy())
            else:
                # none / pin / bid: take first `replan` steps of selected chunk
                for i in range(min(self.replan, pinned.shape[0])):
                    self._queue.append(pinned[i].numpy())

        # Queue holds actions in the SAME space as predict_action_chunk (pre-post).
        # Pin/ensemble math must stay in that space; only then run the postprocessor.
        raw = np.asarray(self._queue.popleft(), dtype=np.float32).reshape(-1)
        self._last_action = torch.as_tensor(raw).float()
        at = self._last_action.unsqueeze(0)  # (1, A)
        try:
            out = self.post(at)
            if torch.is_tensor(out):
                a = out.to("cpu").float().numpy().reshape(-1)
            elif isinstance(out, dict):
                a = out["action"].to("cpu").float().numpy().reshape(-1)
            else:
                a = np.asarray(out, dtype=np.float32).reshape(-1)
        except Exception:
            a = raw
        self._step += 1
        return a


@torch.no_grad()
def rollout_exec(ctrl: ExecController, env, task_description, max_steps, seed):
    ctrl.reset()
    obs, info = env.reset(seed=seed)
    for _ in range(max_steps):
        a = ctrl.act(obs, task_description)
        obs, reward, terminated, truncated, info = env.step(a)
        if info.get("is_success", False):
            return True
        if terminated or truncated:
            break
    return False


def eval_indist_exec(models, mode, replan, n_episodes, max_steps, results, out_path,
                     suite, bench_name, task_ids, decay_tau, ens_alpha,
                     bid_k=8, bid_noise=0.01):
    bench = results.setdefault(bench_name, {})
    for mname, (policy, pre, post) in models.items():
        per_model = bench.setdefault(mname, {"per_task": {}, "average": None, "exec_mode": mode})
        ctrl = ExecController(policy, pre, post, mode=mode, replan=replan,
                              decay_tau=decay_tau, ens_alpha=ens_alpha,
                              bid_k=bid_k, bid_noise=bid_noise)
        for task_id in task_ids:
            key = str(task_id)
            if key in per_model["per_task"]:
                continue
            env = ev._make_base_libero_env(task_id, suite_name=suite)
            task_desc = env.task_description
            succ = [rollout_exec(ctrl, env, task_desc, max_steps, seed=1000 + ep)
                    for ep in range(n_episodes)]
            env.close()
            rate = float(np.mean(succ))
            per_model["per_task"][key] = {
                "task": task_desc, "success_rate": rate, "n": n_episodes,
            }
            print(f"[{bench_name}/{mode}] {mname} task{task_id}: {rate*100:.1f}% "
                  f"({sum(succ)}/{n_episodes})", flush=True)
            ev._save(results, out_path)
        rates = [v["success_rate"] for v in per_model["per_task"].values()]
        per_model["average"] = float(np.mean(rates)) if rates else None
        ev._save(results, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", default="results_exec")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--dataset", default="lerobot/libero_10_image")
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--exec", choices=["none", "pin", "ens", "pinens", "bid"], default="pin")
    ap.add_argument("--indist_episodes", type=int, default=20)
    ap.add_argument("--n_tasks", type=int, default=10)
    ap.add_argument("--task_ids", default=None, help="comma list, e.g. 0,1,2,3,4")
    ap.add_argument("--max_steps", type=int, default=520)
    ap.add_argument("--decay_tau", type=float, default=0.25)
    ap.add_argument("--ens_alpha", type=float, default=0.5)
    ap.add_argument("--bid_k", type=int, default=8)
    ap.add_argument("--bid_noise", type=float, default=0.01)
    args = ap.parse_args()

    task_ids = (list(range(args.n_tasks)) if not args.task_ids
                else [int(x) for x in args.task_ids.split(",") if x.strip() != ""])

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) / "success_rates.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    meta = ev.LeRobotDatasetMetadata(args.dataset)
    models = {}
    for spec in args.model:
        name, head, ckpt = spec.split("=", 2)
        models[name] = ev.load_policy(head, ckpt, meta.stats, args.replan)
        print(f"[exec] loaded {name} head={head} exec={args.exec} replan={args.replan}",
              flush=True)

    bench = args.suite.replace("libero_", "LIBERO-").upper()
    results.setdefault("_config", {})[args.suite] = {
        "replan": args.replan, "exec": args.exec, "episodes": args.indist_episodes,
        "task_ids": task_ids, "decay_tau": args.decay_tau, "ens_alpha": args.ens_alpha,
        "bid_k": args.bid_k, "bid_noise": args.bid_noise,
    }
    eval_indist_exec(models, args.exec, args.replan, args.indist_episodes, args.max_steps,
                     results, out_path, args.suite, bench, task_ids,
                     args.decay_tau, args.ens_alpha,
                     bid_k=args.bid_k, bid_noise=args.bid_noise)
    print("[exec] DONE", json.dumps(results.get(bench, {}), indent=2)[:2000], flush=True)


if __name__ == "__main__":
    main()
