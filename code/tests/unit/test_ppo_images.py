import gymnasium as gym
import numpy as np
import pytest
import torch

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg.schema import PPOConfig
from irl.models.distributions import CategoricalDist, DiagGaussianDist
from irl.models.networks import PolicyNetwork, ValueNetwork


@pytest.mark.parametrize(
    "act_space",
    [
        gym.spaces.Discrete(3),
        gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    ],
    ids=["discrete", "continuous"],
)
def test_gae_and_ppo_update_with_images(act_space):
    obs_space = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    T, B, H, W, C = 3, 2, 32, 32, 3
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 256, size=(T, B, H, W, C), dtype=np.uint8)
    next_obs = rng.integers(0, 256, size=(T, B, H, W, C), dtype=np.uint8)
    rewards = (0.1 * rng.standard_normal(size=(T, B))).astype(np.float32)
    dones = np.zeros((T, B), dtype=np.float32)
    dones[-1] = 1.0

    adv, v_targets = compute_gae(
        {"obs": obs, "next_observations": next_obs, "rewards": rewards, "dones": dones},
        value,
        gamma=0.99,
        lam=0.95,
    )

    N = T * B
    assert adv.shape == (N,)
    assert v_targets.shape == (N,)

    obs_flat = obs.reshape(N, H, W, C)
    with torch.no_grad():
        dist = policy.distribution(obs_flat)
        if hasattr(act_space, "n"):
            assert isinstance(dist, CategoricalDist)
            actions = dist.sample().long().view(N)
        else:
            assert isinstance(dist, DiagGaussianDist)
            actions = dist.sample().view(N, -1)

    cfg = PPOConfig(
        steps_per_update=N,
        minibatches=2,
        epochs=1,
        learning_rate=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
    )

    stats = ppo_update(
        policy,
        value,
        {"obs": obs_flat, "actions": actions},
        adv,
        v_targets,
        cfg,
        return_stats=True,
    )

    assert stats is not None
    for k in ("approx_kl", "clip_frac", "entropy", "policy_loss", "value_loss", "epochs_ran"):
        assert k in stats
        assert np.isfinite(float(stats[k]))
