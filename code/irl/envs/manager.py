"""Build single or vectorized Gymnasium envs with consistent settings.

Includes seeding, optional frame-skip, CarRacing discrete control, and
best-effort domain randomization. Returns a ``gym.Env`` or a ``VectorEnv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from .wrappers import FrameSkip, CarRacingDiscreteActionWrapper, DomainRandomizationWrapper


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
    render_mode :
        Optional render mode forwarded to :func:`gymnasium.make`.
    async_vector :
        If ``True`` and ``num_envs > 1``, use :class:`AsyncVectorEnv`;
        otherwise use :class:`SyncVectorEnv`.
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
        Vec = AsyncVectorEnv if self.async_vector else SyncVectorEnv
        return Vec(thunks, copy=False)

    # ------------------ internal helpers ------------------

    def _make_env_thunk(self, rank: int) -> Callable[[], gym.Env]:
        def thunk() -> gym.Env:
            kwargs = dict(self.make_kwargs or {})
            if self.render_mode is not None:
                kwargs["render_mode"] = self.render_mode

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
                    env = CarRacingDiscreteActionWrapper(env)
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
