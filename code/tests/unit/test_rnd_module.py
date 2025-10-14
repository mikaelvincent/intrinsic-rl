import numpy as np
import gymnasium as gym
import torch

from irl.intrinsic.rnd import RND, RNDConfig
from irl.intrinsic import IntrinsicOutput


def _rand_obs(obs_dim: int, B: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, obs_dim)).astype(np.float32)


def test_rnd_shapes_and_compute_batch():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    rnd = RND(obs_space, device="cpu", cfg=RNDConfig(feature_dim=32, hidden=(64, 64)))

    obs = _rand_obs(6, B=10, seed=1)
    next_obs = _rand_obs(6, B=10, seed=2)

    r1 = rnd.compute_batch(obs)  # uses obs if next_obs None
    r2 = rnd.compute_batch(obs, next_obs)  # prefer next_obs

    assert r1.shape == (10,)
    assert r2.shape == (10,)
    assert torch.isfinite(r1).all()
    assert torch.isfinite(r2).all()

    # Single-transition compute()
    tr = type("T", (), {"s": obs[0], "a": 0, "r_ext": 0.0, "s_next": next_obs[0]})
    out = rnd.compute(tr)
    assert isinstance(out, IntrinsicOutput)
    assert np.isfinite(out.r_int)


def test_rnd_update_reduces_loss_on_fixed_batch():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    cfg = RNDConfig(feature_dim=32, hidden=(64, 64), lr=5e-4)
    rnd = RND(obs_space, device="cpu", cfg=cfg)

    batch = _rand_obs(5, B=64, seed=123)

    # Measure initial loss
    with torch.no_grad():
        initial = float(rnd.loss(batch)["total"])

    # Train predictor on the same batch for a few steps
    for _ in range(10):
        rnd.update(batch)

    with torch.no_grad():
        final = float(rnd.loss(batch)["total"])

    # Expect non-increasing loss on the fixed dataset (stochastic but should drop)
    assert final <= initial or abs(final - initial) < 1e-6
