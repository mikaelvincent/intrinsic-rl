"""Build single or vectorized Gymnasium envs with consistent settings.

Includes seeding, optional frame-skip, CarRacing discrete control, and
best-effort domain randomization. Returns a ``gym.Env`` or a ``VectorEnv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from .wrappers import FrameSkip, CarRacingDiscreteActionWrapper, DomainRandomizationWrapper
from irl.utils.loggers import get_logger

_LOG = get_logger(__name__)


def _is_car_racing(env_id: str) -> bool:
    """Return True if the environment id corresponds to a CarRacing task."""
    return env_id.startswith("CarRacing")


@dataclass
class EnvManager:
    """Factory for single or vectorized Gymnasium environments.

    Parameters
    ----------
    env_id :
        Gymnasium environment id passed to :func:`gymnasium.make`.
    num_envs :
        Number of parallel environments. When ``num_envs == 1`` the
        manager returns a plain :class:`gym.Env`; otherwise it builds a
        vectorised environment.
    seed :
        Base random seed. Each environment receives a deterministic
        offset derived from this seed for reproducible runs.
    frame_skip :
        Number of underlying environment steps to perform for each call
        to :meth:`step`. Values greater than 1 wrap the env in
        :class:`FrameSkip`.
    domain_randomization :
        If ``True``, wrap the env in :class:`DomainRandomizationWrapper`
        to apply small physics perturbations on every reset.
    discrete_actions :
        When ``True`` and ``env_id`` identifies a CarRacing task, wrap
        the env in :class:`CarRacingDiscreteActionWrapper` to expose a
        small discrete action space.
    car_action_set :
        Optional discrete action set specification for CarRacing
        environments. When not ``None`` and ``discrete_actions`` is
        ``True``, it must be array-like of shape ``(N, 3)`` containing
        ``[steer, gas, brake]`` triples and overrides the wrapper's
        default 5-action set.
    render_mode :
        Optional render mode forwarded to :func:`gymnasium.make`.
    async_vector :
        If ``True`` and ``num_envs > 1``, use :class:`AsyncVectorEnv`;
        otherwise use :class:`SyncVectorEnv`. If async creation fails
        (e.g., non-picklable envs), the manager falls back to a sync
        vector env and logs a one-line warning.
    make_kwargs :
        Additional keyword arguments forwarded to
        :func:`gymnasium.make` when constructing the base environment.
    """

    env_id: str
    num_envs: int = 1
    seed: Optional[int] = 1
    frame_skip: int = 1
    domain_randomization: bool = False
    discrete_actions: bool = True
    car_action_set: Optional[object] = None
    render_mode: Optional[str] = None
    async_vector: bool = False
    make_kwargs: Optional[Dict[str, Any]] = None

    # ------------------ public API ------------------

    def make(self):
        """Instantiate the configured environment or vector env.

        Returns
        -------
        gym.Env or gym.vector.VectorEnv
            A single Gymnasium environment when ``num_envs == 1`` or a
            vectorised environment when ``num_envs > 1``.
        """
        thunks = [self._make_env_thunk(rank) for rank in range(self.num_envs)]

        if self.num_envs == 1:
            # Return the raw env instance for the single-env case
            return thunks[0]()  # type: ignore[misc]

        # Vectorized path
        if self.async_vector:
            # Try async first; fall back to sync if construction fails for any reason.
            # IMPORTANT: prefer copy=True (when supported) to avoid shared-buffer reuse
            # that can silently corrupt stored rollouts (especially for image observations).
            try:
                try:
                    return AsyncVectorEnv(thunks, copy=True)  # type: ignore[call-arg]
                except TypeError:
                    return AsyncVectorEnv(thunks)  # older gymnasium versions
            except Exception as exc:
                _LOG.info(
                    "AsyncVectorEnv failed (%s); falling back to SyncVectorEnv for env_id=%s.",
                    type(exc).__name__,
                    self.env_id,
                )

        # Sync vector (default or fallback)
        try:
            return SyncVectorEnv(thunks, copy=True)  # type: ignore[call-arg]
        except TypeError:
            return SyncVectorEnv(thunks)

    # ------------------ internal helpers ------------------

    def _build_carracing_action_set(self) -> Optional[np.ndarray]:
        """Resolve the configured CarRacing action set to an (N, 3) array.

        When ``car_action_set`` is ``None``, returns ``None`` so that the
        wrapper can use its built-in default. Otherwise, accepts any
        nested sequence or NumPy array convertible to shape ``(N, 3)``.
        """
        spec = self.car_action_set
        if spec is None:
            return None
        try:
            arr = np.asarray(spec, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"car_action_set must be array-like with shape (N, 3); got {type(spec).__name__}."
            ) from exc
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"car_action_set must have shape (N, 3) for [steer, gas, brake]; got {arr.shape}."
            )
        return arr

    def _make_env_thunk(self, rank: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            kwargs = dict(self.make_kwargs or {})
            if self.render_mode is not None:
                kwargs["render_mode"] = self.render_mode

            # Force SDL to use the dummy driver if not otherwise configured.
            # This prevents PyGame-based environments (Classic Control, Box2D) from
            # attempting to open a window or spamming "XDG_RUNTIME_DIR not set" in headless setups.
            if "SDL_VIDEODRIVER" not in os.environ:
                os.environ["SDL_VIDEODRIVER"] = "dummy"

            env = gym.make(self.env_id, **kwargs)

            # --- Seeding (deterministic per rank) ---
            if self.seed is not None:
                # Gymnasium envs accept seeding via reset(seed=...), and spaces can be seeded too.
                try:
                    env.reset(seed=int(self.seed) + rank)  # initial deterministic state
                except Exception:
                    # Some envs don't accept seed via reset kwargs; ignore.
                    pass
                try:
                    env.action_space.seed(int(self.seed) + 17 * rank + 13)
                except Exception:
                    pass
                try:
                    env.observation_space.seed(int(self.seed) + 31 * rank + 7)
                except Exception:
                    pass

            # --- Wrappers in a safe, order-aware stack ---
            # 1) CarRacing discrete controls (if requested).
            if _is_car_racing(self.env_id) and self.discrete_actions:
                try:
                    action_set = self._build_carracing_action_set()
                    env = CarRacingDiscreteActionWrapper(env, action_set=action_set)
                except Exception:
                    # If wrapper is incompatible (unexpected action space), skip gracefully.
                    pass

            # 2) Frame skip.
            if self.frame_skip and self.frame_skip > 1:
                env = FrameSkip(env, skip=self.frame_skip)

            # 3) Domain randomization (best-effort).
            if self.domain_randomization:
                env = DomainRandomizationWrapper(
                    env, seed=None if self.seed is None else int(self.seed) + rank
                )

            # 4) Always collect episode statistics for logging later.
            env = RecordEpisodeStatistics(env)

            return env

        return thunk
