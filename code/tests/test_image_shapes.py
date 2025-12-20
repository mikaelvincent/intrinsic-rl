from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from irl.intrinsic.icm import ICM, ICMConfig
from irl.intrinsic.rnd import RND, RNDConfig
from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.utils.images import infer_channels_hw


def test_grayscale_hw_observations_work_end_to_end() -> None:
    c, hw = infer_channels_hw((32, 32))
    assert c == 1
    assert hw == (32, 32)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    H, W = 32, 32
    obs_space = gym.spaces.Box(low=0, high=255, shape=(H, W), dtype=np.uint8)
    act_space = gym.spaces.Discrete(3)

    policy = PolicyNetwork(obs_space, act_space).to(torch.device("cpu"))
    value = ValueNetwork(obs_space).to(torch.device("cpu"))

    obs = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    a = policy.distribution(obs).sample()
    assert a.shape == (1,)
    v = value(obs)
    assert v.shape == (1,)
    assert torch.isfinite(v).all()

    B = 4
    obs_b = rng.integers(0, 256, size=(B, H, W), dtype=np.uint8)
    dist_b = policy.distribution(obs_b)
    assert dist_b.logits.shape == (B, int(act_space.n))
    v_b = value(obs_b)
    assert v_b.shape == (B,)
    assert torch.isfinite(v_b).all()

    icm = ICM(obs_space, act_space, device="cpu", cfg=ICMConfig(phi_dim=32, hidden=(64, 64)))
    next_obs_b = rng.integers(0, 256, size=(B, H, W), dtype=np.uint8)
    actions = rng.integers(0, int(act_space.n), size=(B,), endpoint=False, dtype=np.int64)

    r_icm = icm.compute_batch(obs_b, next_obs_b, actions, reduction="none")
    assert r_icm.shape == (B,)
    assert torch.isfinite(r_icm).all()
    assert torch.isfinite(icm.loss(obs_b, next_obs_b, actions)["intrinsic_mean"]).all()

    rnd = RND(obs_space, device="cpu", cfg=RNDConfig(feature_dim=32, hidden=(64, 64)))
    r_rnd = rnd.compute_batch(obs_b, reduction="none")
    assert r_rnd.shape == (B,)
    assert torch.isfinite(r_rnd).all()
