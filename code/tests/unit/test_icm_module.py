import numpy as np
import gymnasium as gym
import torch

from irl.intrinsic.icm import ICM, ICMConfig
from irl.intrinsic import IntrinsicOutput


def _rand_batch(obs_dim, act_space, B=8, seed=0):
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


def test_icm_discrete_shapes_and_update():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    icm = ICM(obs_space, act_space, device="cpu", cfg=ICMConfig(phi_dim=32, hidden=(64, 64)))
    obs, next_obs, acts = _rand_batch(4, act_space, B=16, seed=1)

    # Forward compute_batch
    r = icm.compute_batch(obs, next_obs, acts)  # [B]
    assert r.shape == (16,)
    assert torch.isfinite(r).all()

    # Loss/Update
    losses = icm.loss(obs, next_obs, acts)
    for k in ["total", "forward", "inverse", "intrinsic_mean"]:
        assert k in losses
        assert torch.isfinite(losses[k])

    metrics = icm.update(obs, next_obs, acts, steps=2)
    for v in metrics.values():
        assert np.isfinite(v)

    # Single-transition compute()
    tr = type("T", (), {"s": obs[0], "a": int(acts[0]), "r_ext": 0.0, "s_next": next_obs[0]})
    out = icm.compute(tr)
    assert isinstance(out, IntrinsicOutput)
    assert np.isfinite(out.r_int)


def test_icm_continuous_shapes_and_update():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    icm = ICM(obs_space, act_space, device="cpu", cfg=ICMConfig(phi_dim=32, hidden=(64, 64)))
    obs, next_obs, acts = _rand_batch(5, act_space, B=12, seed=2)

    r = icm.compute_batch(obs, next_obs, acts)  # [B]
    assert r.shape == (12,)
    assert torch.isfinite(r).all()

    losses = icm.loss(obs, next_obs, acts)
    for k in ["total", "forward", "inverse", "intrinsic_mean"]:
        assert k in losses
        assert torch.isfinite(losses[k])

    metrics = icm.update(obs, next_obs, acts)
    for v in metrics.values():
        assert np.isfinite(v)

    # Single compute
    tr = type("T", (), {"s": obs[0], "a": acts[0], "r_ext": 0.0, "s_next": next_obs[0]})
    out = icm.compute(tr)
    assert np.isfinite(out.r_int)
