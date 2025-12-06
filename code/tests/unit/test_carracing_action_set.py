import numpy as np
import gymnasium as gym

from irl.envs.wrappers import CarRacingDiscreteActionWrapper


class _DummyCarEnv(gym.Env):
    """Minimal CarRacing-like env for testing the discrete wrapper.

    - Observation: uint8 RGB image (H, W, C).
    - Action: Box(3,) with steering, gas, brake.
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(32, 32, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


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
