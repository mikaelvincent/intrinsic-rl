import gymnasium as gym
import numpy as np
import torch

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg.schema import PPOConfig
from irl.models.distributions import DiagGaussianDist
from irl.models.networks import PolicyNetwork, ValueNetwork


def test_policy_value_support_vector_continuous():
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    rng = np.random.default_rng(0)
    batch = rng.standard_normal((8, 11)).astype(np.float32)

    with torch.no_grad():
        dist = policy.distribution(batch)
        assert isinstance(dist, DiagGaussianDist)

        a = dist.sample()
        lp = dist.log_prob(a)
        ent = dist.entropy()
        v = value(batch)

    assert a.shape == (batch.shape[0], act_space.shape[0])
    assert lp.shape == (batch.shape[0],)
    assert ent.shape == (batch.shape[0],)
    assert v.shape == (batch.shape[0],)
    assert torch.isfinite(v).all()
    assert torch.isfinite(lp).all()
    assert torch.isfinite(ent).all()


def test_gae_and_ppo_update_continuous():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    T, B, D = 3, 4, 7
    rng = np.random.default_rng(1)
    obs = rng.standard_normal((T, B, D)).astype(np.float32)
    next_obs = rng.standard_normal((T, B, D)).astype(np.float32)
    rewards = (0.1 * rng.standard_normal((T, B))).astype(np.float32)
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

    obs_flat = obs.reshape(N, D)
    with torch.no_grad():
        actions = policy.distribution(obs_flat).sample()

    ppo_cfg = PPOConfig(
        steps_per_update=N,
        minibatches=4,
        epochs=1,
        learning_rate=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
    )
    ppo_update(policy, value, {"obs": obs_flat, "actions": actions}, adv, v_targets, ppo_cfg)
