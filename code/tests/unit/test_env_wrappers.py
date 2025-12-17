import gymnasium as gym
import numpy as np
import pytest

from irl.envs.wrappers import CarRacingDiscreteActionWrapper, FrameSkip


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


def test_carracing_action_wrapper_maps_actions():
    env = _DummyCarEnv()
    try:
        wrapped = CarRacingDiscreteActionWrapper(env)
        assert wrapped.action_space.n == 5
    finally:
        wrapped.close()

    custom = [
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
    ]
    env = _DummyCarEnv()
    try:
        wrapped = CarRacingDiscreteActionWrapper(env, action_set=custom)
        assert wrapped.action_space.n == len(custom)
        for i, row in enumerate(custom):
            assert np.allclose(wrapped.action(i), np.asarray(row, dtype=np.float32))
        with pytest.raises(ValueError):
            wrapped.action(len(custom))
    finally:
        wrapped.close()


class _DummyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        obs = np.array([float(self.t)], dtype=np.float32)
        reward = 1.0
        terminated = self.t >= 3
        truncated = False
        info = {"t": self.t}
        return obs, reward, terminated, truncated, info


def test_frameskip_accumulates_and_stops_early():
    env = _DummyEnv()
    try:
        env = FrameSkip(env, skip=2)
        obs, _ = env.reset()
        assert obs.shape == (1,)

        obs1, r1, term1, trunc1, _ = env.step(0)
        assert np.isclose(r1, 2.0)
        assert not term1 and not trunc1
        assert np.isclose(obs1[0], 2.0)

        obs2, r2, term2, trunc2, _ = env.step(1)
        assert np.isclose(r2, 1.0)
        assert term2 and not trunc2
        assert np.isclose(obs2[0], 3.0)
    finally:
        env.close()
