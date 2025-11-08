import numpy as np
import torch
import gymnasium as gym

from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg.schema import PPOConfig


def test_gae_and_ppo_update_with_images():
    # Image obs; discrete actions
    obs_space = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
    act_space = gym.spaces.Discrete(3)

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    # Build small time-major batch of images: (T, B, H, W, C)
    T, B, H, W, C = 3, 2, 32, 32, 3
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 256, size=(T, B, H, W, C), dtype=np.uint8)
    next_obs = rng.integers(0, 256, size=(T, B, H, W, C), dtype=np.uint8)
    rewards = (0.1 * rng.standard_normal(size=(T, B))).astype(np.float32)
    dones = np.zeros((T, B), dtype=np.float32)
    dones[-1] = 1.0

    gae_batch = {"obs": obs, "next_observations": next_obs, "rewards": rewards, "dones": dones}
    adv, v_targets = compute_gae(gae_batch, value, gamma=0.99, lam=0.95)

    N = T * B
    assert adv.shape == (N,) and v_targets.shape == (N,)

    # Flatten obs for PPO while keeping image dims (N, H, W, C)
    obs_flat = obs.reshape(N, H, W, C)
    with torch.no_grad():
        dist = policy.distribution(obs_flat)
        actions = dist.sample().long().view(N)

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
