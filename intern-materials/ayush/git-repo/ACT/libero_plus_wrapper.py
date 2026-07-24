"""
libero_plus_wrapper.py
======================
Thin access layer over the LIBERO-Plus benchmark (sylvestf/LIBERO-plus) for
closed-loop robustness evaluation of SmolVLA-family policies.

LIBERO-Plus is a drop-in `libero` package whose perturbations (camera viewpoint,
lighting, sensor noise, background textures, object layout, robot initial state,
language) are baked into per-task BDDL/init-state files and decoded by its
OffScreenRenderEnv. We therefore construct envs strictly through the benchmark
API (get_task_bddl_file_path + get_task_init_states) so every perturbation is
applied exactly as the benchmark intends.

Isolation: LIBERO-Plus is selected via sys.path + a dedicated LIBERO_CONFIG_PATH
(~/.libero_plus) so it never clashes with the original LIBERO used elsewhere.
Import this module BEFORE importing `libero`.
"""

from __future__ import annotations

import os
import sys
import json

# --- isolate LIBERO-Plus from the original LIBERO (must precede `import libero`)
_PLUS_ROOT = "/home/user/Desktop/Ayush PH test/third_party/LIBERO-plus"
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_plus"))
os.environ.setdefault("MUJOCO_GL", "egl")
if _PLUS_ROOT not in sys.path:
    sys.path.insert(0, _PLUS_ROOT)

import numpy as np
# NumPy 2.0 removed these aliases; LIBERO-Plus's image-corruption code (e.g. the
# `fog` perturbation -> plasma_fractal) still uses np.float_. Restore them.
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
import torch as _torch

# LIBERO-Plus's get_task_init_states() uses torch.load on pickled numpy arrays.
# PyTorch >=2.6 defaults weights_only=True, which rejects them. The benchmark
# files are from a trusted source (the official assets we downloaded), so restore
# the legacy behaviour for these internal loads.
_orig_torch_load = _torch.load
def _compat_torch_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
_torch.load = _compat_torch_load

from libero.libero import benchmark as _benchmark
from libero.libero.envs import OffScreenRenderEnv

_CLS_PATH = os.path.join(_PLUS_ROOT, "libero/libero/benchmark/task_classification.json")

# The 7 LIBERO-Plus perturbation dimensions (category strings as in classification).
CATEGORIES = [
    "Camera Viewpoints",
    "Light Conditions",
    "Sensor Noise",
    "Background Textures",
    "Objects Layout",
    "Robot Initial States",
    "Language Instructions",
]


def get_benchmark(suite: str = "libero_spatial"):
    return _benchmark.get_benchmark_dict()[suite]()


# Perturbation keyword that begins the suffix LIBERO-Plus appends to a task name
# for the non-language categories (camera/light/texture/layout/noise/init). When a
# task has no explicit language field, LIBERO-Plus derives the instruction from the
# (perturbed) name -> the instruction ends up like "... in the basket view 0 0 100
# 2 4 initstate 0". Feeding that to a language-conditioned policy is off-distribution
# and tanks success ("false 0%"). For the Language Instructions category the language
# field is a genuine rephrase and MUST be kept verbatim.
def clean_instruction(raw: str, clean_set, category: str | None = None) -> str:
    """Recover the clean training instruction from a (possibly suffix-corrupted) raw
    instruction by longest-prefix match against the dataset's training instructions.
    The Language Instructions category is a real rephrase -> returned unchanged.
    Falls back to `raw` if nothing matches (also leaves Language rephrases intact)."""
    if category == "Language Instructions" or not clean_set:
        return raw
    r = raw.strip()
    best = None
    for s in clean_set:
        s = s.strip()
        if r == s or (r.startswith(s) and r[len(s):len(s) + 1] in (" ", "")):
            if best is None or len(s) > len(best):
                best = s
    return best if best is not None else raw


def list_perturbed_tasks(suite: str = "libero_spatial"):
    """Return (benchmark_instance, [task_dicts]) where each dict has
    {id, name, category, difficulty_level, index} and `index` is the benchmark
    task index usable with get_task_* APIs."""
    cls = json.load(open(_CLS_PATH))[suite]
    bench = get_benchmark(suite)
    name_to_idx = {n: i for i, n in enumerate(bench.get_task_names())}
    out = []
    for c in cls:
        if c["name"] in name_to_idx:
            out.append({**c, "index": name_to_idx[c["name"]]})
    return bench, out


class LiberoPlusEnv:
    """Closed-loop env for a single perturbed LIBERO-Plus task index."""

    def __init__(self, bench, index: int, img_size: int = 256,
                 category: str | None = None, clean_instructions=None):
        self.bench = bench
        self.index = index
        bddl = bench.get_task_bddl_file_path(index)
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=img_size, camera_widths=img_size
        )
        self.init_states = bench.get_task_init_states(index)
        task = bench.get_task(index)
        raw = getattr(task, "language", None) or \
            getattr(self.env, "language_instruction", "")
        # For non-language perturbations LIBERO-Plus name-derives the instruction and
        # appends the perturbation suffix (e.g. "... view 0 0 100 2 4 initstate 0").
        # Recover the clean training instruction; keep genuine Language rephrases.
        self.task_description = clean_instruction(raw, clean_instructions, category)

    def reset(self, seed: int = 0, settle_steps: int = 5):
        obs = self.env.reset()
        # LIBERO-Plus bakes each perturbation into the BDDL, so env.reset() already
        # spawns the PERTURBED scene. The colleague's harness passes init_states=False
        # for exactly this reason: calling set_init_state can re-inject a base-task qpos
        # that mismatches the perturbed BDDL. Toggle via LP_SKIP_SET_INIT=1 to compare.
        if os.environ.get("LP_SKIP_SET_INIT", "0") != "1":
            st = self.init_states[seed % len(self.init_states)]
            obs = self.env.set_init_state(st)
        # Match the OFFICIAL LIBERO eval protocol (lifelong/evaluate.py:264-265,
        # metric.py:119-122): step the sim with zero actions so the scene SETTLES
        # before the policy acts. Right after set_init_state the objects are still
        # interpenetrating / falling and the gripper is in a transient pose, so the
        # first observations are off-distribution. Skipping this makes grasping
        # tasks (Object suite) collapse to ~0%. Applied identically to all variants.
        for _ in range(settle_steps):
            obs, _, _, _ = self.env.step(np.zeros(7, dtype=np.float32))
        return obs

    def step(self, action):
        return self.env.step(action)  # (obs, reward, done, info)

    def check_success(self) -> bool:
        return bool(self.env.check_success())

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Quick API probe (no policy): enumerate tasks per category for a suite.
    import collections
    bench, tasks = list_perturbed_tasks("libero_spatial")
    print("n benchmark tasks:", bench.n_tasks, "| classified+matched:", len(tasks))
    cat = collections.Counter(t["category"] for t in tasks)
    for k, v in cat.most_common():
        print(f"  {k:25s} {v}")
