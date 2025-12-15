import gymnasium as gym
import numpy as np
import torch

from irl.intrinsic import IntrinsicOutput
from irl.intrinsic.rnd import RND, RNDConfig


def _rand_obs(obs_dim: int, B: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, obs_dim)).astype(np.float32)


def test_rnd_shapes_and_compute_batch():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    rnd = RND(obs_space, device="cpu", cfg=RNDConfig(feature_dim=32, hidden=(64, 64)))

    obs = _rand_obs(6, B=10, seed=1)
    next_obs = _rand_obs(6, B=10, seed=2)

    r1 = rnd.compute_batch(obs)
    r2 = rnd.compute_batch(obs, next_obs)

    assert r1.shape == (10,)
    assert r2.shape == (10,)
    assert torch.isfinite(r1).all()
    assert torch.isfinite(r2).all()

    tr = type("T", (), {"s": obs[0], "a": 0, "r_ext": 0.0, "s_next": next_obs[0]})
    out = rnd.compute(tr)
    assert isinstance(out, IntrinsicOutput)
    assert np.isfinite(out.r_int)


def test_rnd_update_reduces_loss_on_fixed_batch():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    cfg = RNDConfig(feature_dim=32, hidden=(64, 64), lr=5e-4)
    rnd = RND(obs_space, device="cpu", cfg=cfg)

    batch = _rand_obs(5, B=64, seed=123)

    with torch.no_grad():
        initial = float(rnd.loss(batch)["total"])

    for _ in range(10):
        rnd.update(batch)

    with torch.no_grad():
        final = float(rnd.loss(batch)["total"])

    assert final <= initial or abs(final - initial) < 1e-6


def test_rnd_compute_batch_updates_rms_when_normalized():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    cfg = RNDConfig(
        feature_dim=16,
        hidden=(32, 32),
        lr=3e-4,
        rms_beta=0.0,
        normalize_intrinsic=True,
    )
    rnd = RND(obs_space, device="cpu", cfg=cfg)

    obs = _rand_obs(6, B=8, seed=42)

    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32)
        per = rnd._intrinsic_raw_per_sample(x)
        r2_expected = float((per**2).mean().item())

    r = rnd.compute_batch(obs)
    assert r.shape == (8,)
    assert torch.isfinite(r).all()

    r2_ema = float(rnd._r2_ema.item())
    assert abs(r2_ema - r2_expected) < 1e-6
