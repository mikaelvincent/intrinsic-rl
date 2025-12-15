import gymnasium as gym
import numpy as np
import torch

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg.schema import PPOConfig
from irl.models.networks import PolicyNetwork, ValueNetwork


def test_gae_and_ppo_update_shapes_and_smoke():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    T, B, D = 4, 3, 4
    obs = np.random.randn(T, B, D).astype(np.float32)
    next_obs = np.roll(obs, shift=-1, axis=0)
    rewards = (0.1 * np.random.randn(T, B)).astype(np.float32)
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
    assert torch.isfinite(adv).all() and torch.isfinite(v_targets).all()

    obs_flat = torch.as_tensor(obs.reshape(N, D), dtype=torch.float32)
    with torch.no_grad():
        actions = policy.distribution(obs_flat).sample().long().view(N)

    ppo_cfg = PPOConfig(
        steps_per_update=N,
        minibatches=2,
        epochs=1,
        learning_rate=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
    )
    ppo_update(policy, value, {"obs": obs_flat, "actions": actions}, adv, v_targets, ppo_cfg)
