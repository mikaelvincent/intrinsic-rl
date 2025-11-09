"""Common environment wrappers used by the project."""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FrameSkip(gym.Wrapper):
    """Repeat the same action for `skip` steps and sum rewards.

    Stops early if a termination/truncation occurs. Returns the last obs/info.
    """

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
    """Convert CarRacing's Box(3,) actions into a small Discrete set.

    Default actions: no-op, left, right, gas, brake (5).
    """

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
        if not isinstance(action_set, np.ndarray) or action_set.shape[1:] != (3,):
            raise ValueError("action_set must be a NumPy array of shape (N, 3).")
        self._action_set = action_set.astype(np.float32)
        self.action_space = spaces.Discrete(self._action_set.shape[0])

    def action(self, act: int):
        """Map a discrete index to the underlying Box(3,) action with bounds-checking."""
        try:
            idx = int(act)
        except Exception as exc:
            raise ValueError(
                f"Discrete action must be an integer index (got {act!r})."
            ) from exc

        n = int(self._action_set.shape[0])
        if idx < 0 or idx >= n:
            raise ValueError(
                f"Action index {idx} is out of bounds for action set of size {n}. "
                f"Valid range is [0, {n - 1}]."
            )
        return self._action_set[idx]


class DomainRandomizationWrapper(gym.Wrapper):
    """Apply small physics perturbations where available (best-effort).

    * MuJoCo: scale gravity (~±5%) and geom friction (~±10%).
    * Box2D : scale vertical gravity (~±5%).
    Randomization is applied before every `reset`.
    """

    def __init__(self, env: gym.Env, seed: Optional[int] = None) -> None:
        super().__init__(env)
        # Use numpy RNG; vector environments can still be reproducible via per-env seeding.
        self._rng = np.random.default_rng(seed)
        self._last_applied = 0  # diagnostics only

    # ------------------ helpers ------------------

    def _u(self, low: float, high: float, size=None):
        return self._rng.uniform(low, high, size=size)

    def _maybe_randomize_mujoco(self) -> int:
        applied = 0
        try:
            uw = self.env.unwrapped
            mj_model = getattr(uw, "model", None) or getattr(uw, "mujoco_model", None)
            if mj_model is None:
                return 0

            # Gravity (vector of length 3)
            try:
                scale = self._u(0.95, 1.05)
                g = np.array(mj_model.opt.gravity, dtype=np.float64)
                g *= scale
                mj_model.opt.gravity[:] = g
                applied += 1
            except Exception:
                pass

            # Geom friction (N geoms x 3) if present
            try:
                fr = np.array(mj_model.geom_friction, dtype=np.float64, copy=True)
                noise = self._u(0.9, 1.1, size=fr.shape)
                mj_model.geom_friction[:] = fr * noise
                applied += 1
            except Exception:
                pass
        except Exception:
            # Silent: best-effort only
            return applied
        return applied

    def _maybe_randomize_box2d(self) -> int:
        applied = 0
        try:
            uw = self.env.unwrapped
            world = getattr(uw, "world", None)
            if world is None:
                return 0
            g = getattr(world, "gravity", None)
            # Box2D gravity may be tuple-like (x, y) or object with x/y
            if g is not None:
                scale = self._u(0.95, 1.05)
                try:
                    # tuple-like
                    gx, gy = g
                    world.gravity = (gx, gy * scale)
                    applied += 1
                except Exception:
                    try:
                        # attribute-like
                        world.gravity = (world.gravity[0], world.gravity[1] * scale)
                        applied += 1
                    except Exception:
                        pass
        except Exception:
            return applied
        return applied

    def _apply_randomization(self) -> None:
        applied = 0
        applied += self._maybe_randomize_mujoco()
        applied += self._maybe_randomize_box2d()
        self._last_applied = applied

    # ------------------ gym API ------------------

    def reset(self, **kwargs):
        self._apply_randomization()
        return self.env.reset(**kwargs)
