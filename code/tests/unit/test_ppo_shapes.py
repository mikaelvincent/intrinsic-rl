import gymnasium as gym
import numpy as np
import torch

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg.schema import PPOConfig
from irl.models.networks import PolicyNetwork, ValueNetwork


def _rand_vector(T: int, B: int, D: int, seed: int):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((T, B, D)).astype(np.float32)
    next_obs = rng.standard_normal((T, B, D)).astype(np.float32)
    rewards = (0.1 * rng.standard_normal((T, B))).astype(np.float32)
    dones = np.zeros((T, B), dtype=np.float32)
    dones[-1] = 1.0
    return obs, next_obs, rewards, dones


def _rand_images(T: int, B: int, H: int, W: int, C: int, seed: int):
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 256, size=(T, B, H, W, C), dtype=np.uint8)
    next_obs = rng.integers(0, 256, size=(T, B, H, W, C), dtype=np.uint8)
    rewards = (0.1 * rng.standard_normal((T, B))).astype(np.float32)
    dones = np.zeros((T, B), dtype=np.float32)
    dones[-1] = 1.0
    return obs, next_obs, rewards, dones


def _run_gae_and_ppo(
    obs_space: gym.Space,
    act_space: gym.Space,
    obs,
    next_obs,
    rewards,
    dones,
    obs_flat,
):
    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    adv, v_targets = compute_gae(
        {"obs": obs, "next_observations": next_obs, "rewards": rewards, "dones": dones},
        value,
        gamma=0.99,
        lam=0.95,
    )

    N = int(obs_flat.shape[0])
    assert adv.shape == (N,)
    assert v_targets.shape == (N,)
    assert torch.isfinite(adv).all()
    assert torch.isfinite(v_targets).all()

    with torch.no_grad():
        dist = policy.distribution(obs_flat)
        actions = dist.sample()
        if hasattr(act_space, "n"):
            actions = actions.long().view(N)
        else:
            actions = actions.view(N, -1)

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
        assert np.isfinite(float(stats[k]))
    assert float(stats["epochs_ran"]) == 1.0


def test_gae_and_ppo_vector_discrete():
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    T, B, D = 4, 2, 4
    obs, next_obs, rewards, dones = _rand_vector(T, B, D, seed=0)
    obs_flat = obs.reshape(T * B, D)

    _run_gae_and_ppo(obs_space, act_space, obs, next_obs, rewards, dones, obs_flat)


def test_gae_and_ppo_image_continuous():
    obs_space = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    T, B, H, W, C = 4, 2, 32, 32, 3
    obs, next_obs, rewards, dones = _rand_images(T, B, H, W, C, seed=0)
    obs_flat = obs.reshape(T * B, H, W, C)

    _run_gae_and_ppo(obs_space, act_space, obs, next_obs, rewards, dones, obs_flat)
