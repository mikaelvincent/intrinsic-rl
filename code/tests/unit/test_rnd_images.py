import numpy as np
import gymnasium as gym
import torch

from irl.intrinsic.rnd import RND, RNDConfig


def _rand_images(B=6, H=32, W=32, C=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(B, H, W, C), dtype=np.uint8)


def test_rnd_supports_image_observations():
    # Image Box obs; arbitrary CarRacing-like shape
    obs_space = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
    cfg = RNDConfig(feature_dim=32, hidden=(64, 64))  # hidden unused for images

    rnd = RND(obs_space, device="cpu", cfg=cfg)

    obs = _rand_images(B=5, H=32, W=32, C=3, seed=1)
    next_obs = _rand_images(B=5, H=32, W=32, C=3, seed=2)

    r1 = rnd.compute_batch(obs)  # uses obs if next_obs None
    r2 = rnd.compute_batch(obs, next_obs)

    assert r1.shape == (5,)
    assert r2.shape == (5,)
    assert torch.isfinite(r1).all()
    assert torch.isfinite(r2).all()
