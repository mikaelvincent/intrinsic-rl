import numpy as np
import gymnasium as gym

from irl.envs.wrappers import FrameSkip


class _DummyEnv(gym.Env):
    """Tiny deterministic env to validate FrameSkip behavior.

    - Observation: single float (time step counter).
    - Reward: +1 per underlying `step` call.
    - Episode terminates at t >= 3.
    """

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
        obs, info = env.reset()
        assert obs.shape == (1,)

        # First call: two underlying steps (t: 0->1->2); not done yet.
        obs1, r1, term1, trunc1, info1 = env.step(0)
        assert np.isclose(r1, 2.0)
        assert not term1 and not trunc1
        assert np.isclose(obs1[0], 2.0)

        # Second call: first inner step reaches termination (t: 2->3), wrapper stops early.
        obs2, r2, term2, trunc2, info2 = env.step(1)
        assert np.isclose(r2, 1.0)  # only one underlying step executed
        assert term2 and not trunc2
        assert np.isclose(obs2[0], 3.0)
    finally:
        env.close()
