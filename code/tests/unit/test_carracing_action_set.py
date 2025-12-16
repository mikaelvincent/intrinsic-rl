import gymnasium as gym
import numpy as np
import pytest

from irl.envs.wrappers import CarRacingDiscreteActionWrapper


class _DummyCarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        return obs, 0.0, False, False, {}


def test_carracing_default_action_set_has_five_actions():
    env = _DummyCarEnv()
    wrapped = CarRacingDiscreteActionWrapper(env)
    try:
        assert wrapped.action_space.n == 5
    finally:
        wrapped.close()


def test_carracing_custom_action_set_maps_indices():
    custom = [
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
    ]

    env = _DummyCarEnv()
    wrapped = CarRacingDiscreteActionWrapper(env, action_set=custom)
    try:
        assert wrapped.action_space.n == len(custom)
        for i, row in enumerate(custom):
            assert np.allclose(wrapped.action(i), np.asarray(row, dtype=np.float32))

        with pytest.raises(ValueError):
            wrapped.action(len(custom))
    finally:
        wrapped.close()
