import gymnasium as gym
import numpy as np
import pytest
import torch

from irl.intrinsic.icm import ICM, ICMConfig


def _rand_batch(obs_dim: int, act_space: gym.Space, B: int, seed: int):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    next_obs = rng.standard_normal((B, obs_dim)).astype(np.float32)

    if isinstance(act_space, gym.spaces.Discrete):
        acts = rng.integers(0, act_space.n, size=(B,), endpoint=False, dtype=np.int64)
    else:
        low = np.where(np.isfinite(act_space.low), act_space.low, -1.0)
        high = np.where(np.isfinite(act_space.high), act_space.high, 1.0)
        acts = rng.uniform(low, high, size=(B, act_space.shape[0])).astype(np.float32)

    return obs, next_obs, acts


@pytest.mark.parametrize(
    "act_space",
    [
        gym.spaces.Discrete(3),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    ],
    ids=["discrete", "continuous"],
)
def test_icm_compute_batch_matches_loss_mean(act_space):
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    icm = ICM(obs_space, act_space, device="cpu", cfg=ICMConfig(phi_dim=32, hidden=(64, 64)))

    B = 32
    obs, next_obs, acts = _rand_batch(5, act_space, B=B, seed=1)

    r = icm.compute_batch(obs, next_obs, acts, reduction="none")
    assert r.shape == (B,)
    assert torch.isfinite(r).all()

    losses = icm.loss(obs, next_obs, acts)
    assert torch.isfinite(losses["total"])
    assert torch.allclose(losses["intrinsic_mean"], r.mean(), atol=1e-5)

    metrics = icm.update(obs, next_obs, acts, steps=1)
    for k in ("loss_total", "loss_forward", "loss_inverse", "intrinsic_mean"):
        assert np.isfinite(float(metrics[k]))
