import numpy as np
import gymnasium as gym
import torch

from irl.intrinsic.proposed import Proposed
from irl.intrinsic.icm import ICMConfig


def test_proposed_gate_rate_metric_smoke():
    # Small vector observation/action spaces
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    # Lightweight nets for speed
    mod = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
    )

    rng = np.random.default_rng(0)
    B = 32
    obs = rng.standard_normal((B, 4)).astype(np.float32)
    next_obs = rng.standard_normal((B, 4)).astype(np.float32)
    actions = rng.integers(0, 3, size=(B,), endpoint=False, dtype=np.int64)

    # Populate regions/stats
    r = mod.compute_batch(obs, next_obs, actions)
    assert r.shape == (B,)
    assert torch.isfinite(r).all()

    # gate_rate should be a valid fraction in [0, 1]
    rate = mod.gate_rate
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0
