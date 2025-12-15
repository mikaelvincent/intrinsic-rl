import gymnasium as gym
import numpy as np
import torch

from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.proposed import Proposed


def test_proposed_gate_rate_metric_smoke():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)
    mod = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
    )

    rng = np.random.default_rng(0)
    obs = rng.standard_normal((32, 4)).astype(np.float32)
    next_obs = rng.standard_normal((32, 4)).astype(np.float32)
    actions = rng.integers(0, 3, size=(32,), endpoint=False, dtype=np.int64)

    r = mod.compute_batch(obs, next_obs, actions)
    assert r.shape == (32,)
    assert torch.isfinite(r).all()

    rate = mod.gate_rate
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0
