"""
eval_groot.py — GR00T comp-3 closed-loop LIBERO eval CLIENT (runs in the PROJECT venv).

Drives the sim (original LIBERO for in-dist, LIBERO-Plus for robustness) and queries a
GR00T PolicyServer for actions. The server is launched separately in the gr00t conda env
(serve_groot.sh) with the GR00T_ACTION_HEAD matching the checkpoint's head — otherwise the
trained head is silently discarded (env-driven head reconstruction, verified).

Why a custom client (not gr00t's rollout_policy): the Plus benchmark has no native GR00T
path, and the project venv (not the gr00t env) is the one that has robosuite + original
LIBERO + LIBERO-Plus. We reuse ONE obs/action code path for in-dist and Plus so control
frequency + transforms are identical across the two (the replan-consistency lesson).

Obs + gripper handling replicate gr00t/eval/sim/LIBERO/libero_env.py EXACTLY:
  * images: raw robosuite HWC uint8, 180-deg flipped [::-1,::-1], keys video.image/wrist_image
  * state:  per-DOF keys state.x/y/z (eef_pos), state.roll/pitch/yaw (quat2axisangle(eef_quat)),
            state.gripper (robot0_gripper_qpos, 2-dim)
  * action: server returns delta 6-DoF + gripper; apply normalize_gripper_action THEN
            invert_gripper_action, feed straight to the delta OSC_POSE controller.
Every video/state entry carries a (B=1, T=1) prefix (sim-policy-wrapper flat format).

Isolation: original LIBERO and LIBERO-Plus both own the top-level `libero` module, so
--only indist and --only plus MUST run in separate interpreters. --only both re-invokes
this script twice and merges, exactly like eval_pi05.py.

Contract: $RESULTS/<variant>__<suite>.json = {comparison, variant, suite, indist_avg, plus_avg}.
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np

# torch>=2.6 defaults torch.load(weights_only=True), which rejects the numpy globals in
# LIBERO's local .pruned_init init-state files -> get_task_init_states() raises
# UnpicklingError. These are trusted local files; restore the pre-2.6 behavior so both the
# in-dist (original LIBERO) and Plus paths can load their init states. Patch at import time,
# before any `from libero...` (which happens lazily inside the env builders below).
import torch as _torch
_torch_load_orig = _torch.load
def _torch_load_compat(*a, **k):
    k.setdefault("weights_only", False)
    return _torch_load_orig(*a, **k)
_torch.load = _torch_load_compat

REPO = "/home/user/Desktop/Ayush PH test"
V2 = os.path.join(REPO, "DeepONet PH", "v2")
GR00T = "/home/user/Isaac-GR00T"

N_TASKS = 10
SUITE_MAP = {"Spatial": "libero_spatial", "Object": "libero_object",
             "Long": "libero_10", "Goal": "libero_goal"}   # NB: Long -> libero_10
CATEGORIES = ["Camera Viewpoints", "Light Conditions", "Sensor Noise", "Background Textures",
              "Objects Layout", "Robot Initial States", "Language Instructions"]


# ------------------------------------------------------------------ transforms (copied verbatim)
def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action


def process_obs(obs, task_desc):
    """Raw robosuite obs -> GR00T sim-policy-wrapper flat dict with (B=1,T=1) prefix."""
    xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)

    def vid(a):
        return np.asarray(a[::-1, ::-1], dtype=np.uint8)[None, None]      # (1,1,H,W,C)

    def st(v):
        return np.asarray(v, dtype=np.float32).reshape(1, 1, -1)          # (1,1,D)

    return {
        "video.image": vid(obs["agentview_image"]),
        "video.wrist_image": vid(obs["robot0_eye_in_hand_image"]),
        "state.x": st([xyz[0]]), "state.y": st([xyz[1]]), "state.z": st([xyz[2]]),
        "state.roll": st([rpy[0]]), "state.pitch": st([rpy[1]]), "state.yaw": st([rpy[2]]),
        "state.gripper": st(grip),
        "annotation.human.action.task_description": [task_desc],          # (B,)
    }


_ACT_KEYS = ["action.x", "action.y", "action.z", "action.roll",
             "action.pitch", "action.yaw", "action.gripper"]


def action_chunk(resp):
    """Server action dict -> (H,7) array of per-step [x,y,z,roll,pitch,yaw,gripper] deltas."""
    if isinstance(resp, (tuple, list)):
        resp = resp[0]
    cols = [np.asarray(resp[k], dtype=np.float32).reshape(-1) for k in _ACT_KEYS]
    H = min(len(c) for c in cols)
    return np.stack([np.array([c[t] for c in cols], dtype=np.float32) for t in range(H)], axis=0)


# ------------------------------------------------------------------ client
def make_client(host, port):
    if GR00T not in sys.path:
        sys.path.insert(0, GR00T)
    from gr00t.policy.server_client import PolicyClient
    c = PolicyClient(host=host, port=port)
    assert c.ping(), f"GR00T server not reachable at {host}:{port}"
    return c


def query(client, obs):
    # low-level call skips client-side strict validation; the server's sim-policy-wrapper
    # validates + converts the flat dict to the model's nested modality format.
    return client._get_action(obs)


# ------------------------------------------------------------------ envs (raw robosuite, settle + delta)
class _RawEnv:
    """Shared reset(settle + use_delta) / step / success wrapper over an OffScreenRenderEnv,
    matching comp-1's LiberoPlusEnv reset protocol so in-dist and Plus behave identically."""
    def __init__(self, env, init_states, task_description):
        self.env = env
        self.init_states = init_states
        self.task_description = task_description

    def reset(self, seed=0, num_steps_wait=10):
        self.env.reset()
        if self.init_states is not None and len(self.init_states):
            self.env.set_init_state(self.init_states[seed % len(self.init_states)])
        obs = None
        for _ in range(num_steps_wait):                       # physics-settle
            obs, _, _, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
        for robot in self.env.robots:                         # delta OSC controller
            robot.controller.use_delta = True
        return obs

    def step(self, a):
        return self.env.step(a)

    def check_success(self):
        return bool(self.env.check_success())

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def indist_env(suite_key, task_id):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    bench = benchmark.get_benchmark_dict()[suite_key]()
    bddl = bench.get_task_bddl_file_path(task_id)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    init_states = bench.get_task_init_states(task_id)
    task = bench.get_task(task_id)
    return _RawEnv(env, init_states, getattr(task, "language", "") or "")


# ------------------------------------------------------------------ rollout
def rollout(client, env, task_desc, max_steps, horizon, seed=0):
    try:
        client.reset()
    except Exception:
        pass
    obs = env.reset(seed=seed)
    steps = 0
    while steps < max_steps:
        chunk = action_chunk(query(client, process_obs(obs, task_desc)))
        n = min(horizon, chunk.shape[0])
        for t in range(n):
            vec = chunk[t].copy()
            vec = normalize_gripper_action(vec)
            vec = invert_gripper_action(vec)
            obs, _, done, _ = env.step(vec)
            steps += 1
            if env.check_success():
                return True
            if done or steps >= max_steps:
                return False
    return False


# ------------------------------------------------------------------ in-dist / plus drivers
def run_indist(args, client, scratch):
    suite_key = SUITE_MAP[args.suite]
    out = {"suite": suite_key, "per_task": {}}
    rates = []
    for task_id in range(N_TASKS):
        env = indist_env(suite_key, task_id)
        succ = []
        for ep in range(args.indist_episodes):      # vary init state per episode via seed
            succ.append(rollout(client, env, env.task_description, args.max_steps,
                                args.horizon, seed=ep))
        env.close()
        rate = float(np.mean(succ))
        out["per_task"][str(task_id)] = {"task": env.task_description, "success_rate": rate,
                                         "n": args.indist_episodes}
        rates.append(rate)
        (scratch / "indist.json").write_text(json.dumps(out, indent=2))
        print(f"[groot/indist] {args.suite} task{task_id}: {rate*100:.1f}%", flush=True)
    return float(np.mean(rates)) if rates else None


def run_plus(args, client, scratch):
    sys.path.insert(0, V2)
    import libero_plus_wrapper as LP  # noqa: F401  (MUST precede libero import; sets isolation)
    from libero_plus_wrapper import LiberoPlusEnv, list_perturbed_tasks, CATEGORIES as CATS
    from evaluate_plus import stratified_sample   # defined in evaluate_plus, NOT libero_plus_wrapper
    suite_key = SUITE_MAP[args.suite]
    bench, tasks = list_perturbed_tasks(suite_key)
    by_cat = {c: [t for t in tasks if t["category"] == c] for c in CATS}
    sampled = {c: stratified_sample(by_cat[c], args.plus_episodes, seed=42) for c in CATS}
    per_cat, cat_avgs = {}, []
    for c in CATS:
        vals = []
        for t in sampled[c]:
            penv = LiberoPlusEnv(bench, t["index"], img_size=256)   # has reset(seed)/step/check_success
            vals.append(rollout(client, penv, penv.task_description, args.max_steps,
                                args.horizon, seed=0))
            penv.close()
        avg = float(np.mean(vals)) if vals else None
        per_cat[c] = avg
        if avg is not None:
            cat_avgs.append(avg)
        (scratch / "robustness_plus.json").write_text(
            json.dumps({"per_category": per_cat}, indent=2))
        print(f"[groot/plus] {args.suite} {c}: {avg}", flush=True)
    rob = float(np.mean(cat_avgs)) if cat_avgs else None
    (scratch / "robustness_plus.json").write_text(
        json.dumps({"per_category": per_cat, "robustness_average": rob}, indent=2))
    return rob


# ------------------------------------------------------------------ orchestration
def _partial(results, variant, suite, only):
    return Path(results) / f"_partial_groot_{variant}__{suite}__{only}.json"


def run_single(args):
    scratch = Path(args.results) / f"_{args.variant}__{args.suite}_raw"
    scratch.mkdir(parents=True, exist_ok=True)
    client = make_client(args.host, args.port)
    if args.only == "indist":
        val, key = run_indist(args, client, scratch), "indist_avg"
    else:
        val, key = run_plus(args, client, scratch), "plus_avg"
    _partial(args.results, args.variant, args.suite, args.only).write_text(json.dumps({key: val}))
    print(f"[eval_groot:{args.only}] {args.variant}/{args.suite} {key}={val}", flush=True)


def run_both(args):
    base = [sys.executable, os.path.abspath(__file__),
            "--variant", args.variant, "--suite", args.suite, "--results", args.results,
            "--host", args.host, "--port", str(args.port),
            "--indist_episodes", str(args.indist_episodes), "--plus_episodes", str(args.plus_episodes),
            "--max_steps", str(args.max_steps), "--horizon", str(args.horizon)]
    vals = {"indist_avg": None, "plus_avg": None}
    for only in ("indist", "plus"):
        rc = subprocess.run(base + ["--only", only]).returncode
        p = _partial(args.results, args.variant, args.suite, only)
        if rc == 0 and p.exists():
            vals.update(json.loads(p.read_text()))
            p.unlink()
        else:
            print(f"[eval_groot] WARNING: {only} rc={rc} (result left null)", flush=True)
    # FAIL LOUDLY on a null metric: a crashed in-dist/plus subprocess must NOT look like
    # success. We do NOT write the contract JSON on failure (so the runner keeps the
    # checkpoint, retries the suite, and never feeds nulls into aggregation), and we exit 1.
    if vals["indist_avg"] is None or vals["plus_avg"] is None:
        print(f"[eval_groot] FAIL {args.variant}/{args.suite}: null metric "
              f"(indist={vals['indist_avg']} plus={vals['plus_avg']}) — contract JSON NOT written",
              flush=True)
        sys.exit(1)
    out = {"comparison": "c3", "variant": args.variant, "suite": args.suite,
           "indist_avg": vals["indist_avg"], "plus_avg": vals["plus_avg"]}
    Path(args.results, f"{args.variant}__{args.suite}.json").write_text(json.dumps(out, indent=2))
    print(f"[eval_groot] {args.variant}/{args.suite} indist={vals['indist_avg']} plus={vals['plus_avg']}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--suite", required=True, choices=list(SUITE_MAP))
    ap.add_argument("--results", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--indist_episodes", type=int, default=12)
    ap.add_argument("--plus_episodes", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=520)
    ap.add_argument("--horizon", type=int, default=8, help="action-chunk steps executed per query")
    ap.add_argument("--only", choices=["indist", "plus", "both"], default="both")
    args = ap.parse_args()
    Path(args.results).mkdir(parents=True, exist_ok=True)
    (run_both if args.only == "both" else run_single)(args)


if __name__ == "__main__":
    main()
