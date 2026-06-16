"""
libero_v_wrapper.py
===================
LIBERO-V: a visual-perturbation wrapper around standard LIBERO, implemented with
MuJoCo's camera / light / image APIs (used when an official `libero_v` package is
not available -- see .libero_v_backend written by setup_blackwell.sh).

Three perturbation types, applied on reset() and held fixed for the episode so
each rollout is deterministic given its seed:

  1. "viewpoint"     : +/-15 deg camera viewpoint shift.
                       Implemented as a RIGID world rotation about the camera's
                       look-at point, applied to BOTH camera position and
                       orientation. This yields an exact angular viewpoint change
                       while keeping the workspace centered -- no look-at
                       quaternion fitting needed.
  2. "lighting"      : directional light intensity + position/direction change
                       (scales model.light_diffuse / shifts model.light_pos,dir).
  3. "sensor_noise"  : additive Gaussian noise on the rendered RGB images.

Built on lerobot.envs.libero.LiberoEnv (gym.Env). The underlying MuJoCo sim is
reached via  env._env.env.sim .

Usage
-----
    from libero_v_wrapper import make_libero_v_env
    env = make_libero_v_env(task_id=0, perturbation="viewpoint", severity=15.0, seed=0)
    obs, info = env.reset()
    ...
    # clean (no perturbation) env for the in-distribution comparison:
    env = make_libero_v_env(task_id=0, perturbation="none")

Self-test (needs LIBERO + EGL):
    MUJOCO_GL=egl python libero_v_wrapper.py --selftest
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

PERTURBATIONS = ("none", "viewpoint", "lighting", "sensor_noise")


# --------------------------------------------------------------------------- quats
# MuJoCo quaternion convention is [w, x, y, z].
def axisangle_to_quat(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    h = 0.5 * angle_rad
    return np.array([np.cos(h), *(np.sin(h) * axis)], dtype=np.float64)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_rotate(q, v):
    """Rotate 3-vector v by quaternion q ([w,x,y,z])."""
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return R @ np.asarray(v, dtype=np.float64)


# --------------------------------------------------------------------------- factory
def _make_base_libero_env(task_id, suite_name="libero_10", **kwargs):
    """Instantiate lerobot's LiberoEnv for a given LIBERO-10 task."""
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    defaults = dict(obs_type="pixels_agent_pos", control_mode="relative",
                    observation_height=256, observation_width=256)
    defaults.update(kwargs)
    return LiberoEnv(task_suite=suite, task_suite_name=suite_name,
                     task_id=task_id, **defaults)


def make_libero_v_env(task_id, perturbation="none", severity=None, seed=0,
                      suite_name="libero_10", **kwargs):
    """Return a LiberoVWrapper around LiberoEnv for one task + perturbation type."""
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"perturbation must be one of {PERTURBATIONS}, got {perturbation}")
    base = _make_base_libero_env(task_id, suite_name=suite_name, **kwargs)
    return LiberoVWrapper(base, perturbation=perturbation, severity=severity, seed=seed)


# --------------------------------------------------------------------------- wrapper
class LiberoVWrapper(gym.Wrapper):
    # sensible default severities per type
    DEFAULTS = {"viewpoint": 15.0,        # degrees
                "lighting": 0.6,          # diffuse scale spread (+/-)
                "sensor_noise": 0.06}     # gaussian std as fraction of 255

    def __init__(self, env, perturbation="none", severity=None, seed=0):
        super().__init__(env)
        self.perturbation = perturbation
        self.severity = self.DEFAULTS.get(perturbation, 0.0) if severity is None else severity
        self._rng = np.random.default_rng(seed)
        self._noise_active = False  # set on reset for sensor_noise

    # -- access to underlying MuJoCo sim --------------------------------------
    @property
    def _sim(self):
        return self.env._env.env.sim

    def _cam_id(self, name="agentview"):
        try:
            return self._sim.model.camera_name2id(name)
        except Exception:
            import mujoco
            return mujoco.mj_name2id(self._sim.model._model,
                                     mujoco.mjtObj.mjOBJ_CAMERA, name)

    # -- perturbation implementations -----------------------------------------
    def _apply_viewpoint(self):
        sim = self._sim
        cid = self._cam_id("agentview")
        pos = np.array(sim.model.cam_pos[cid], dtype=np.float64)
        quat = np.array(sim.model.cam_quat[cid], dtype=np.float64)

        # Camera looks down its local -Z. Estimate the look-at pivot ~1 m ahead.
        forward = quat_rotate(quat, [0.0, 0.0, -1.0])
        pivot = pos + forward * 1.0

        # signed +/- severity degrees about world-Z (azimuth viewpoint shift)
        sign = 1.0 if self._rng.random() < 0.5 else -1.0
        ang = np.deg2rad(sign * float(self.severity))
        Rw = axisangle_to_quat([0.0, 0.0, 1.0], ang)

        new_pos = pivot + quat_rotate(Rw, pos - pivot)
        new_quat = quat_mul(Rw, quat)
        sim.model.cam_pos[cid] = new_pos
        sim.model.cam_quat[cid] = new_quat / (np.linalg.norm(new_quat) + 1e-12)
        sim.forward()

    def _apply_lighting(self):
        sim = self._sim
        nlight = int(sim.model.nlight)
        if nlight == 0:
            return
        spread = float(self.severity)
        for i in range(nlight):
            # intensity scale in [1-spread, 1+spread]
            scale = 1.0 + self._rng.uniform(-spread, spread)
            sim.model.light_diffuse[i] = np.clip(
                np.array(sim.model.light_diffuse[i]) * scale, 0.0, 1.0)
            # shift position and re-aim direction a little
            sim.model.light_pos[i] = np.array(sim.model.light_pos[i]) + \
                self._rng.uniform(-0.5, 0.5, size=3)
            d = np.array(sim.model.light_dir[i]) + self._rng.uniform(-0.3, 0.3, size=3)
            n = np.linalg.norm(d)
            if n > 1e-6:
                sim.model.light_dir[i] = d / n
        sim.forward()

    def _add_noise(self, obs):
        std = float(self.severity) * 255.0
        pix = obs.get("pixels", {})
        for k, img in pix.items():
            if isinstance(img, np.ndarray) and img.ndim == 3:
                noisy = img.astype(np.float32) + self._rng.normal(0, std, img.shape)
                pix[k] = np.clip(noisy, 0, 255).astype(np.uint8)
        obs["pixels"] = pix
        return obs

    # -- re-render obs after a model edit -------------------------------------
    def _refetch_obs(self):
        # force_update=True is REQUIRED: robosuite caches observations from the
        # last sim step, so without it the camera/light edits would not show up.
        base = self.env
        raw = base._env.env._get_observations(force_update=True)
        return base._format_raw_obs(raw)

    # -- gym API --------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._noise_active = False
        if self.perturbation == "viewpoint":
            self._apply_viewpoint()
            obs = self._refetch_obs()
        elif self.perturbation == "lighting":
            self._apply_lighting()
            obs = self._refetch_obs()
        elif self.perturbation == "sensor_noise":
            self._noise_active = True
            obs = self._add_noise(obs)
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        # gymnasium 5-tuple (obs, reward, terminated, truncated, info)
        obs = out[0]
        if self._noise_active:
            obs = self._add_noise(obs)
        return (obs, *out[1:])


# --------------------------------------------------------------------------- self-test
def _selftest():
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    print("[libero_v] self-test: render clean vs each perturbation (task 0)")

    def first_image(o):
        return np.asarray(o["pixels"]["image"])

    # One env at a time (create -> use -> close) to avoid EGL multi-context issues.
    clean = make_libero_v_env(0, "none", seed=0)
    o0, _ = clean.reset(seed=0)
    base_img = first_image(o0).astype(np.float32)
    print("  clean frame:", base_img.shape, "mean", round(float(base_img.mean()), 2))
    assert base_img.sum() > 0, "clean frame is black"
    clean.close()

    for pert in ("viewpoint", "lighting", "sensor_noise"):
        env = make_libero_v_env(0, pert, seed=1)
        o, _ = env.reset(seed=0)
        img = first_image(o).astype(np.float32)
        diff = float(np.abs(img - base_img).mean())
        print(f"  {pert:12s} frame mean={img.mean():7.2f}  |Δ vs clean|={diff:7.3f}")
        assert img.sum() > 0, f"{pert} frame is black"
        assert diff > 1.0, f"{pert} did not visibly change the image (Δ={diff:.3f})"
        env.close()
    print("[libero_v] SELF-TEST PASSED")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Run with --selftest (needs LIBERO + EGL). Importable API: make_libero_v_env(...)")
