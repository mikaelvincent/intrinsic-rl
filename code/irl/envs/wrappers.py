from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 2) -> None:
        assert skip >= 1, "skip must be >= 1"
        super().__init__(env)
        self.skip = int(skip)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        last_info = {}
        obs = None
        for _ in range(self.skip):
            obs, r, term, trunc, info = self.env.step(action)
            total_reward += float(r)
            terminated = terminated or bool(term)
            truncated = truncated or bool(trunc)
            last_info = info
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, last_info


class CarRacingDiscreteActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        action_set: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box) or env.action_space.shape != (3,):
            raise TypeError(
                "CarRacingDiscreteActionWrapper expects an env with Box(3,) action space."
            )

        if action_set is None:
            action_set = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        else:
            action_set = np.asarray(action_set, dtype=np.float32)

        if action_set.ndim != 2 or action_set.shape[1:] != (3,):
            raise ValueError("action_set must be an array-like of shape (N, 3).")
        self._action_set = action_set.astype(np.float32)
        self.action_space = spaces.Discrete(self._action_set.shape[0])

    def action(self, act: int):
        try:
            idx = int(act)
        except Exception as exc:
            raise ValueError(f"Discrete action must be an integer index (got {act!r}).") from exc

        n = int(self._action_set.shape[0])
        if idx < 0 or idx >= n:
            raise ValueError(
                f"Action index {idx} is out of bounds for action set of size {n}. "
                f"Valid range is [0, {n - 1}]."
            )
        return self._action_set[idx]


class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: Optional[int] = None) -> None:
        super().__init__(env)
        self._rng = np.random.default_rng(seed)
        self._last_applied = 0
        self._last_diag: dict[str, int] = {"mujoco": 0, "box2d": 0}

        self._backend_checked: bool = False
        self._has_mujoco: bool = False
        self._has_box2d: bool = False

        # Cache baseline physics so noise doesn't drift across resets.
        self._mj_baseline_gravity: np.ndarray | None = None
        self._mj_baseline_geom_friction: np.ndarray | None = None
        self._b2d_baseline_gravity: tuple[float, float] | None = None

    def _u(self, low: float, high: float, size=None):
        return self._rng.uniform(low, high, size=size)

    def _detect_backends(self) -> None:
        if self._backend_checked:
            return
        try:
            uw = self.env.unwrapped
        except Exception:
            self._backend_checked = True
            return

        try:
            mj_model = getattr(uw, "model", None) or getattr(uw, "mujoco_model", None)
            if (
                mj_model is not None
                and hasattr(mj_model, "opt")
                and hasattr(mj_model.opt, "gravity")
            ):
                g = getattr(mj_model.opt, "gravity", None)
                if g is not None:
                    arr = np.array(g, dtype=np.float64).reshape(-1)
                    if arr.size == 3 and np.isfinite(arr).all():
                        self._has_mujoco = True
                        if self._mj_baseline_gravity is None:
                            self._mj_baseline_gravity = arr.copy()
                        if self._mj_baseline_geom_friction is None:
                            fr = getattr(mj_model, "geom_friction", None)
                            if fr is not None:
                                fr_arr = np.array(fr, dtype=np.float64, copy=True)
                                if fr_arr.ndim == 2 and fr_arr.shape[1] == 3 and fr_arr.size > 0:
                                    self._mj_baseline_geom_friction = fr_arr.copy()
        except Exception:
            self._has_mujoco = False

        try:
            world = getattr(uw, "world", None)
            if world is not None and hasattr(world, "gravity"):
                self._has_box2d = True
                if self._b2d_baseline_gravity is None:
                    g = getattr(world, "gravity", None)
                    if g is not None:
                        try:
                            gx, gy = g
                            self._b2d_baseline_gravity = (float(gx), float(gy))
                        except Exception:
                            if hasattr(g, "x") and hasattr(g, "y"):
                                self._b2d_baseline_gravity = (float(g.x), float(g.y))
        except Exception:
            self._has_box2d = False

        self._backend_checked = True

    def _maybe_randomize_mujoco(self) -> int:
        if not self._has_mujoco:
            return 0
        applied = 0
        try:
            uw = self.env.unwrapped
            mj_model = getattr(uw, "model", None) or getattr(uw, "mujoco_model", None)
            if mj_model is None or not hasattr(mj_model, "opt"):
                return 0

            try:
                if self._mj_baseline_gravity is not None and hasattr(mj_model.opt, "gravity"):
                    base_g = np.asarray(self._mj_baseline_gravity, dtype=np.float64).reshape(-1)
                    if base_g.size == 3:
                        mj_model.opt.gravity[:] = base_g
                if self._mj_baseline_geom_friction is not None:
                    base_fr = np.asarray(self._mj_baseline_geom_friction, dtype=np.float64).copy()
                    if base_fr.ndim == 2 and base_fr.shape[1] == 3 and base_fr.size > 0:
                        mj_model.geom_friction[:] = base_fr
            except Exception:
                pass

            try:
                if self._mj_baseline_gravity is not None:
                    base_g = np.asarray(self._mj_baseline_gravity, dtype=np.float64).reshape(-1)
                    if base_g.size == 3 and hasattr(mj_model.opt, "gravity"):
                        scale = self._u(0.95, 1.05)
                        mj_model.opt.gravity[:] = base_g * scale
                        applied += 1
                else:
                    g = getattr(mj_model.opt, "gravity", None)
                    if g is not None:
                        g_arr = np.array(g, dtype=np.float64).reshape(-1)
                        if g_arr.size == 3 and np.isfinite(g_arr).all():
                            scale = self._u(0.95, 1.05)
                            g_arr *= scale
                            mj_model.opt.gravity[:] = g_arr
                            applied += 1
            except Exception:
                pass

            try:
                fr = getattr(mj_model, "geom_friction", None)
                if fr is not None:
                    if self._mj_baseline_geom_friction is not None:
                        base_fr = np.asarray(
                            self._mj_baseline_geom_friction, dtype=np.float64, copy=True
                        )
                        if base_fr.ndim == 2 and base_fr.shape[1] == 3 and base_fr.size > 0:
                            noise = self._u(0.9, 1.1, size=base_fr.shape)
                            mj_model.geom_friction[:] = base_fr * noise
                            applied += 1
                    else:
                        fr_arr = np.array(fr, dtype=np.float64, copy=True)
                        if fr_arr.ndim == 2 and fr_arr.shape[1] == 3 and fr_arr.size > 0:
                            noise = self._u(0.9, 1.1, size=fr_arr.shape)
                            mj_model.geom_friction[:] = fr_arr * noise
                            applied += 1
            except Exception:
                pass
        except Exception:
            return applied
        return applied

    def _maybe_randomize_box2d(self) -> int:
        if not self._has_box2d:
            return 0
        applied = 0
        try:
            uw = self.env.unwrapped
            world = getattr(uw, "world", None)
            if world is None or not hasattr(world, "gravity"):
                return 0

            scale = self._u(0.95, 1.05)

            try:
                if self._b2d_baseline_gravity is not None:
                    gx, gy = self._b2d_baseline_gravity
                else:
                    g = getattr(world, "gravity", None)
                    if g is None:
                        return 0
                    try:
                        gx, gy = g
                    except Exception:
                        gx = float(getattr(g, "x"))
                        gy = float(getattr(g, "y"))
                world.gravity = (float(gx), float(gy) * scale)
                applied += 1
            except Exception:
                try:
                    if self._b2d_baseline_gravity is not None:
                        gx, gy = self._b2d_baseline_gravity
                    else:
                        g = getattr(world, "gravity", None)
                        if g is None:
                            return applied
                        gx = float(getattr(g, "x"))
                        gy = float(getattr(g, "y"))
                    new_gy = gy * scale
                    try:
                        world.gravity = (gx, new_gy)
                        applied += 1
                    except Exception:
                        if hasattr(world.gravity, "x") and hasattr(world.gravity, "y"):
                            world.gravity.x = gx
                            world.gravity.y = new_gy
                            applied += 1
                except Exception:
                    pass
        except Exception:
            return applied
        return applied

    def _apply_randomization(self) -> None:
        self._detect_backends()
        applied_mj = self._maybe_randomize_mujoco()
        applied_b2d = self._maybe_randomize_box2d()
        self._last_diag = {"mujoco": int(applied_mj), "box2d": int(applied_b2d)}
        self._last_applied = int(applied_mj + applied_b2d)

    def reset(self, **kwargs):
        self._apply_randomization()
        obs, info = self.env.reset(**kwargs)
        if not isinstance(info, dict):
            info = {}
        else:
            info = dict(info)  # Copy to avoid mutating upstream containers.
        info.setdefault("dr_applied", dict(self._last_diag))
        return obs, info
