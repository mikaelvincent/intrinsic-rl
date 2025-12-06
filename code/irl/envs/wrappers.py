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
        """Step the environment, repeating the same action for ``skip`` frames.

        The wrapper accumulates rewards across the repeated inner steps and
        stops early if a termination or truncation signal is observed. The
        observation and info dict from the last inner step are returned.
        """
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
            raise ValueError(f"Discrete action must be an integer index (got {act!r}).") from exc

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
    Randomization is applied **before** every `reset()`.

    Diagnostics:
      - Adds an entry to the `reset()` info dict:
          info["dr_applied"] = {"mujoco": <int>, "box2d": <int>}
        indicating how many sub-perturbations were applied per backend.

    Robustness:
      - Safely **detects** supported backends once and avoids mutating when
        the expected structures are not present.
      - Extra shape/type checks prevent accidental writes to unexpected fields.
    """

    def __init__(self, env: gym.Env, seed: Optional[int] = None) -> None:
        super().__init__(env)
        # Use numpy RNG; vector environments can still be reproducible via per-env seeding.
        self._rng = np.random.default_rng(seed)
        self._last_applied = 0  # total diagnostics only (sum)
        self._last_diag: dict[str, int] = {"mujoco": 0, "box2d": 0}  # detailed diagnostics

        # Backend detection flags (evaluated lazily on first reset)
        self._backend_checked: bool = False
        self._has_mujoco: bool = False
        self._has_box2d: bool = False

    # ------------------ helpers ------------------

    def _u(self, low: float, high: float, size=None):
        return self._rng.uniform(low, high, size=size)

    def _detect_backends(self) -> None:
        """Detect whether the wrapped env exposes MuJoCo or Box2D structures.

        This is a no-op after the first successful detection attempt.
        """
        if self._backend_checked:
            return
        try:
            uw = self.env.unwrapped
        except Exception:
            # Cannot introspect; mark checked to avoid repeated attempts.
            self._backend_checked = True
            return

        # MuJoCo: look for .model or .mujoco_model with .opt.gravity at least
        try:
            mj_model = getattr(uw, "model", None) or getattr(uw, "mujoco_model", None)
            if (
                mj_model is not None
                and hasattr(mj_model, "opt")
                and hasattr(mj_model.opt, "gravity")
            ):
                g = getattr(mj_model.opt, "gravity", None)
                # Expect 3-vector gravity; tolerate array-likes
                if g is not None:
                    arr = np.array(g, dtype=np.float64).reshape(-1)
                    if arr.size == 3 and np.isfinite(arr).all():
                        self._has_mujoco = True
        except Exception:
            self._has_mujoco = False  # remain conservative

        # Box2D: look for .world with .gravity attribute
        try:
            world = getattr(uw, "world", None)
            if world is not None and hasattr(world, "gravity"):
                self._has_box2d = True
        except Exception:
            self._has_box2d = False

        self._backend_checked = True

    def _maybe_randomize_mujoco(self) -> int:
        """Best-effort MuJoCo perturbations; returns count of applied changes."""
        if not self._has_mujoco:
            return 0
        applied = 0
        try:
            uw = self.env.unwrapped
            mj_model = getattr(uw, "model", None) or getattr(uw, "mujoco_model", None)
            if mj_model is None or not hasattr(mj_model, "opt"):
                return 0

            # Gravity (vector of length 3)
            try:
                g = getattr(mj_model.opt, "gravity", None)
                if g is not None:
                    g_arr = np.array(g, dtype=np.float64).reshape(-1)
                    if g_arr.size == 3 and np.isfinite(g_arr).all():
                        scale = self._u(0.95, 1.05)
                        g_arr *= scale
                        # Assign back via slice to avoid replacing the object
                        mj_model.opt.gravity[:] = g_arr
                        applied += 1
            except Exception:
                pass

            # Geom friction (N geoms x 3) if present
            try:
                fr = getattr(mj_model, "geom_friction", None)
                if fr is not None:
                    fr_arr = np.array(fr, dtype=np.float64, copy=True)
                    if fr_arr.ndim == 2 and fr_arr.shape[1] == 3 and fr_arr.size > 0:
                        noise = self._u(0.9, 1.1, size=fr_arr.shape)
                        mj_model.geom_friction[:] = fr_arr * noise
                        applied += 1
            except Exception:
                pass
        except Exception:
            # Silent: best-effort only
            return applied
        return applied

    def _maybe_randomize_box2d(self) -> int:
        """Best-effort Box2D perturbations; returns count of applied changes."""
        if not self._has_box2d:
            return 0
        applied = 0
        try:
            uw = self.env.unwrapped
            world = getattr(uw, "world", None)
            if world is None or not hasattr(world, "gravity"):
                return 0

            g = getattr(world, "gravity", None)
            if g is None:
                return 0

            scale = self._u(0.95, 1.05)

            # world.gravity can be tuple-like or a vector type. Handle both.
            try:
                gx, gy = g  # tuple-like
                new_g = (gx, float(gy) * scale)
                world.gravity = new_g
                applied += 1
            except Exception:
                # Attribute-like (e.g., b2Vec2 with .x/.y)
                try:
                    gx = float(getattr(world.gravity, "x"))
                    gy = float(getattr(world.gravity, "y"))
                    # Many Box2D bindings accept tuple assignment to .gravity too
                    try:
                        world.gravity = (gx, gy * scale)
                        applied += 1
                    except Exception:
                        # Fall back to in-place attribute update if exposed
                        if hasattr(world.gravity, "x") and hasattr(world.gravity, "y"):
                            world.gravity.x = gx
                            world.gravity.y = gy * scale
                            applied += 1
                except Exception:
                    pass
        except Exception:
            return applied
        return applied

    def _apply_randomization(self) -> None:
        """Apply DR and update diagnostics (per-backend + total)."""
        # Detect available backends once to avoid poking unknown internals repeatedly
        self._detect_backends()

        applied_mj = self._maybe_randomize_mujoco()
        applied_b2d = self._maybe_randomize_box2d()
        self._last_diag = {"mujoco": int(applied_mj), "box2d": int(applied_b2d)}
        self._last_applied = int(applied_mj + applied_b2d)

    # ------------------ gym API ------------------

    def reset(self, **kwargs):
        """Reset the environment after applying domain randomization.

        Randomization is applied before delegating to the wrapped
        ``reset`` method. The returned info dict is copied (to avoid
        mutating upstream containers) and always contains a
        ``"dr_applied"`` entry with per-backend diagnostics for
        MuJoCo and Box2D, even when no perturbations were applied.
        """
        self._apply_randomization()
        obs, info = self.env.reset(**kwargs)
        # Ensure info is a dict we can enrich with diagnostics
        if not isinstance(info, dict):
            info = {}
        else:
            info = dict(info)  # copy to avoid mutating upstream containers
        # Inject per-backend diagnostics; remain no-op if both zeros
        info.setdefault("dr_applied", dict(self._last_diag))
        return obs, info
