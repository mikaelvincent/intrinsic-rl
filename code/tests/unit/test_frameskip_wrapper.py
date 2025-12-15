import gymnasium as gym
import numpy as np

from irl.envs.wrappers import FrameSkip


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
