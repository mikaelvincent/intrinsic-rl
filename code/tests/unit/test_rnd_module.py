import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F

from irl.intrinsic.rnd import RND, RNDConfig


def _rand_obs(obs_dim: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, obs_dim)).astype(np.float32)


def _rand_images(B: int, H: int, W: int, C: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(B, H, W, C), dtype=np.uint8)


def test_rnd_next_obs_and_rms_update():
    obs_space_img = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
    rnd_img = RND(obs_space_img, device="cpu", cfg=RNDConfig(feature_dim=32, hidden=(64, 64)))

    B = 10
    H, W, C = (int(x) for x in obs_space_img.shape)
    obs = _rand_images(B=B, H=H, W=W, C=C, seed=1)
    next_obs = _rand_images(B=B, H=H, W=W, C=C, seed=2)

    r2 = rnd_img.compute_batch(obs, next_obs)
    r3 = rnd_img.compute_batch(next_obs)

    assert r2.shape == r3.shape == (B,)
    assert torch.isfinite(r2).all()
    assert torch.isfinite(r3).all()
    assert torch.allclose(r2, r3, atol=1e-6)

    obs_space_vec = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    cfg = RNDConfig(feature_dim=16, hidden=(32, 32), rms_beta=0.0, normalize_intrinsic=True)
    rnd = RND(obs_space_vec, device="cpu", cfg=cfg)

    obs2 = _rand_obs(5, B=64, seed=42)

    with torch.no_grad():
        x = torch.as_tensor(obs2, dtype=torch.float32)
        p = rnd.predictor(x)
        tgt = rnd.target(x)
        per = F.mse_loss(p, tgt, reduction="none").mean(dim=-1)
        expected_rms = float(torch.sqrt((per**2).mean() + float(cfg.rms_eps)).item())

    _ = rnd.compute_batch(obs2)
    assert abs(float(rnd.rms) - expected_rms) < 1e-6
