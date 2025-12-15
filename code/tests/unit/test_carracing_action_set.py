import gymnasium as gym
import numpy as np

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
    try:
        wrapped = CarRacingDiscreteActionWrapper(env)
        assert wrapped.action_space.n == 5
    finally:
        wrapped.close()


def test_carracing_custom_action_set_overrides_default():
    env = _DummyCarEnv()
    custom = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.8, 0.2],
        ],
        dtype=np.float32,
    )
    try:
        wrapped = CarRacingDiscreteActionWrapper(env, action_set=custom)
        assert wrapped.action_space.n == custom.shape[0]
        assert np.allclose(wrapped._action_set, custom.astype(np.float32))
    finally:
        wrapped.close()


def test_carracing_action_wrapper_accepts_python_lists():
    env = _DummyCarEnv()
    custom = [
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    try:
        wrapped = CarRacingDiscreteActionWrapper(env, action_set=custom)
        assert wrapped.action_space.n == 3
    finally:
        wrapped.close()
