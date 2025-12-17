from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from irl.utils.loggers import get_logger

from .wrappers import CarRacingDiscreteActionWrapper, DomainRandomizationWrapper, FrameSkip

_LOG = get_logger(__name__)

try:
    from gymnasium.vector import AutoresetMode as _AutoresetMode
except Exception:
    _AutoresetMode = None


def _is_car_racing(env_id: str) -> bool:
    return env_id.startswith("CarRacing")


def _make_vector_env(cls, thunks, *, copy: bool):
    if _AutoresetMode is not None:
        try:
            return cls(thunks, copy=copy, autoreset_mode=_AutoresetMode.SAME_STEP)
        except TypeError:
            pass
    try:
        return cls(thunks, copy=copy, autoreset_mode="same_step")
    except TypeError:
        pass

    if _AutoresetMode is not None:
        try:
            return cls(thunks, autoreset_mode=_AutoresetMode.SAME_STEP)
        except TypeError:
            pass
    try:
        return cls(thunks, autoreset_mode="same_step")
    except TypeError:
        pass

    try:
        return cls(thunks, copy=copy)
    except TypeError:
        return cls(thunks)


@dataclass
class EnvManager:
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

    def make(self):
        thunks = [self._make_env_thunk(rank) for rank in range(self.num_envs)]
        if self.num_envs == 1:
            return thunks[0]()

        if self.async_vector:
            try:
                return _make_vector_env(AsyncVectorEnv, thunks, copy=True)
            except Exception as exc:
                _LOG.info(
                    "AsyncVectorEnv failed (%s); falling back to SyncVectorEnv for env_id=%s.",
                    type(exc).__name__,
                    self.env_id,
                )

        return _make_vector_env(SyncVectorEnv, thunks, copy=True)

    def _build_carracing_action_set(self) -> Optional[np.ndarray]:
        spec = self.car_action_set
        if spec is None:
            return None
        try:
            arr = np.asarray(spec, dtype=np.float32)
        except Exception as exc:
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

            if "SDL_VIDEODRIVER" not in os.environ:
                os.environ["SDL_VIDEODRIVER"] = "dummy"

            env = gym.make(self.env_id, **kwargs)

            if self.seed is not None:
                try:
                    env.action_space.seed(int(self.seed) + 17 * rank + 13)
                except Exception:
                    pass
                try:
                    env.observation_space.seed(int(self.seed) + 31 * rank + 7)
                except Exception:
                    pass

            if _is_car_racing(self.env_id) and self.discrete_actions:
                try:
                    action_set = self._build_carracing_action_set()
                    env = CarRacingDiscreteActionWrapper(env, action_set=action_set)
                except Exception:
                    pass

            if self.frame_skip and self.frame_skip > 1:
                env = FrameSkip(env, skip=self.frame_skip)

            if self.domain_randomization:
                env = DomainRandomizationWrapper(
                    env, seed=None if self.seed is None else int(self.seed) + rank
                )

            env = RecordEpisodeStatistics(env)
            return env

        return thunk
