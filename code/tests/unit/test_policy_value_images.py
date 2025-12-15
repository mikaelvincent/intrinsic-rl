import gymnasium as gym
import numpy as np
import torch

from irl.models.networks import PolicyNetwork, ValueNetwork


def _rand_images(B=4, H=64, W=64, C=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(B, H, W, C), dtype=np.uint8)


def test_policy_value_support_images_discrete():
    obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    act_space = gym.spaces.Discrete(5)

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    batch = _rand_images(B=6, H=64, W=64, C=3, seed=1)
    n = batch.shape[0]
    with torch.no_grad():
        dist = policy.distribution(batch)
        a = dist.sample()
        lp = dist.log_prob(a)
        ent = dist.entropy()
        v = value(batch)

    assert a.shape[0] == n
    assert lp.shape == (n,)
    assert ent.shape == (n,)
    assert v.shape == (n,)
    assert torch.isfinite(v).all()


def test_policy_value_support_images_continuous():
    obs_space = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    batch = _rand_images(B=5, H=32, W=32, C=3, seed=2)
    n = batch.shape[0]
    with torch.no_grad():
        dist = policy.distribution(batch)
        a = dist.sample()
        lp = dist.log_prob(a)
        ent = dist.entropy()
        v = value(batch)

    assert a.shape[0] == n
    assert lp.shape == (n,)
    assert ent.shape == (n,)
    assert v.shape == (n,)
    assert torch.isfinite(v).all()
