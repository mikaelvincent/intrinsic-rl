from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.envs.registration import register

from irl.envs.manager import EnvManager


class _CarRacingLikeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        obs = np.zeros((4,), dtype=np.float32)
        return obs, 0.0, True, False, {}


try:
    register(id="CarRacingLikeStrict-v0", entry_point=_CarRacingLikeEnv)
except Exception:
    pass


def test_carracing_wrapper_failure_raises() -> None:
    mgr = EnvManager(
        env_id="CarRacingLikeStrict-v0",
        num_envs=1,
        seed=0,
        discrete_actions=True,
        car_action_set=[[0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="car_action_set must have shape"):
        _ = mgr.make()
