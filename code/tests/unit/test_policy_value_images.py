import numpy as np
import gymnasium as gym
import torch

from irl.models.networks import PolicyNetwork, ValueNetwork


def _rand_images(B=4, H=64, W=64, C=3, seed=0):
    rng = np.random.default_rng(seed)
    # use uint8 to exercise auto-scaling to [0,1]
    return rng.integers(0, 256, size=(B, H, W, C), dtype=np.uint8)


def test_policy_value_support_images_discrete():
    obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    act_space = gym.spaces.Discrete(5)

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    batch = _rand_images(B=6, H=64, W=64, C=3, seed=1)
    with torch.no_grad():
        # distribution and sampling should work
        dist = policy.distribution(batch)
        a = dist.sample()
        lp = dist.log_prob(a)
        ent = dist.entropy()

        # value should return [B]
        v = value(batch)

    assert a.shape[0] == batch.shape[0]
    assert lp.shape[0] == batch.shape[0]
    assert ent.shape[0] == batch.shape[0]
    assert v.shape[0] == batch.shape[0]
    assert torch.isfinite(v).all()


def test_policy_value_support_images_continuous():
    obs_space = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    batch = _rand_images(B=5, H=32, W=32, C=3, seed=2)
    with torch.no_grad():
        dist = policy.distribution(batch)
        a = dist.sample()
        lp = dist.log_prob(a)
        ent = dist.entropy()
        v = value(batch)

    assert a.shape[0] == batch.shape[0]
    assert lp.shape[0] == batch.shape[0]
    assert ent.shape[0] == batch.shape[0]
    assert v.shape[0] == batch.shape[0]
    assert torch.isfinite(v).all()
