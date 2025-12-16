import gymnasium as gym
import numpy as np
from gymnasium import spaces

from irl.envs.wrappers import DomainRandomizationWrapper


class _DummyMujocoLikeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        class _Model:
            pass

        class _Opt:
            pass

        self.model = _Model()
        self.model.opt = _Opt()
        self.model.opt.gravity = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.model.geom_friction = np.ones((4, 3), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        return obs, 0.0, False, False, {}


def test_domain_randomization_mujoco_stays_near_baseline():
    env = _DummyMujocoLikeEnv()
    try:
        wrapped = DomainRandomizationWrapper(env, seed=123)
        baseline = wrapped.unwrapped.model.opt.gravity.copy()

        unique_scales: set[float] = set()
        for _ in range(30):
            _, info = wrapped.reset()
            g = wrapped.unwrapped.model.opt.gravity
            ratio = g / baseline

            assert np.all(np.isfinite(ratio))
            assert np.all(ratio >= 0.95 - 1e-6)
            assert np.all(ratio <= 1.05 + 1e-6)

            assert isinstance(info, dict)
            diag = info.get("dr_applied")
            assert isinstance(diag, dict)
            assert diag.get("mujoco", 0) >= 0

            unique_scales.add(round(float(ratio.reshape(-1)[0]), 3))

        assert len(unique_scales) > 1
    finally:
        env.close()
