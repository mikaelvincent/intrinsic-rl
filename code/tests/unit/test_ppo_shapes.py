import numpy as np
import torch
import gymnasium as gym

from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg.schema import PPOConfig


def test_gae_and_ppo_update_shapes_and_smoke():
    # Synthetic spaces (Box obs, Discrete actions) to avoid heavy envs
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    # Build a tiny time-major batch: (T, B, obs_dim)
    T, B, D = 4, 3, 4
    obs = np.random.randn(T, B, D).astype(np.float32)
    next_obs = np.roll(obs, shift=-1, axis=0)
    rewards = (0.1 * np.random.randn(T, B)).astype(np.float32)
    dones = np.zeros((T, B), dtype=np.float32)
    dones[-1] = 1.0  # last step terminal

    gae_batch = {
        "obs": obs,
        "next_observations": next_obs,
        "rewards": rewards,
        "dones": dones,
    }

    adv, v_targets = compute_gae(gae_batch, value, gamma=0.99, lam=0.95)
    N = T * B
    assert adv.shape == (N,)
    assert v_targets.shape == (N,)
    assert torch.isfinite(adv).all() and torch.isfinite(v_targets).all()

    # Flatten obs and sample actions from current policy
    obs_flat = torch.as_tensor(obs.reshape(N, D), dtype=torch.float32)
    with torch.no_grad():
        dist = policy.distribution(obs_flat)
        actions = dist.sample().long().view(N)

    # One lightweight PPO update to verify shapes & flow
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
    batch = {"obs": obs_flat, "actions": actions}
    ppo_update(policy, value, batch, adv, v_targets, ppo_cfg)
