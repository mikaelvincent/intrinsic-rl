from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from irl.models.networks import PolicyNetwork
from irl.pipelines.policy_rollout import iter_policy_rollout


class _BoundedBoxEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        lo = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        hi = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)

        assert a.shape == lo.shape == hi.shape
        assert np.all(a >= lo - 1e-6)
        assert np.all(a <= hi + 1e-6)

        self._t += 1
        terminated = self._t >= 3
        obs = np.zeros((4,), dtype=np.float32)
        return obs, 0.0, bool(terminated), False, {}

    def close(self) -> None:
        return


def test_actions_within_bounds_for_box_space() -> None:
    env = _BoundedBoxEnv()
    try:
        obs_space = env.observation_space
        act_space = env.action_space

        policy = PolicyNetwork(obs_space, act_space).to(torch.device("cpu"))
        policy.eval()

        for mode in ("mode", "sample"):
            obs0, _ = env.reset(seed=0)
            for step in iter_policy_rollout(
                env=env,
                policy=policy,
                obs0=obs0,
                act_space=act_space,
                device=torch.device("cpu"),
                policy_mode=mode,
                normalize_obs=None,
                max_steps=5,
            ):
                lp = policy.distribution(step.obs_t).log_prob(step.act_t)
                assert torch.isfinite(lp).all()
    finally:
        env.close()
