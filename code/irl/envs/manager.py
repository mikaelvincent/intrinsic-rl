"""Environment manager that builds single or vectorized Gymnasium envs.

Features:
- Deterministic seeding per environment instance.
- Optional frame-skip wrapper.
- Optional CarRacing discrete actions wrapper (default enabled for CarRacing).
- Optional best-effort domain randomization wrapper.
- Returns either a single `gym.Env` (num_envs == 1) or a `vector.VectorEnv`.

This is a minimal-yet-practical implementation intended for Sprint 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from .wrappers import FrameSkip, CarRacingDiscreteActionWrapper, DomainRandomizationWrapper


def _is_car_racing(env_id: str) -> bool:
    return env_id.startswith("CarRacing")


@dataclass
class EnvManager:
    """Factory for building (vectorized) environments with consistent settings.

    Args:
        env_id: Gymnasium environment id (e.g., "MountainCar-v0", "CarRacing-v3").
        num_envs: Number of parallel envs. If 1, returns a regular `gym.Env`, otherwise a `VectorEnv`.
        seed: Base RNG seed. Each env gets `seed + rank`.
        frame_skip: Repeat actions for `frame_skip` steps (>=1).
        domain_randomization: If True, wrap env with DomainRandomizationWrapper.
        discrete_actions: For CarRacing, if True, use a small Discrete action set wrapper.
        render_mode: Forwarded to `gym.make`. Use "rgb_array" or "human" as supported by the env.
        async_vector: If True and `num_envs>1`, use AsyncVectorEnv instead of SyncVectorEnv.
        make_kwargs: Extra kwargs passed to `gym.make`.
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
        """Instantiate and return the configured environment(s)."""
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
                # gymnasium envs accept seeding via reset(seed=...), and spaces can be seeded too.
                try:
                    env.reset(seed=int(self.seed) + rank)  # initial deterministic state
                except Exception:
                    # Some envs don't accept seed via reset kwargs; ignore
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
            # 1) CarRacing discrete controls (if requested)
            if _is_car_racing(self.env_id) and self.discrete_actions:
                try:
                    env = CarRacingDiscreteActionWrapper(env)
                except Exception:
                    # If wrapper is incompatible (unexpected action space), skip gracefully
                    pass

            # 2) Frame skip
            if self.frame_skip and self.frame_skip > 1:
                env = FrameSkip(env, skip=self.frame_skip)

            # 3) Domain randomization (best-effort)
            if self.domain_randomization:
                env = DomainRandomizationWrapper(
                    env, seed=None if self.seed is None else int(self.seed) + rank
                )

            # 4) Always collect episode statistics for logging later
            env = RecordEpisodeStatistics(env)

            return env

        return thunk
