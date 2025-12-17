from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch

from irl.algo.ppo import ppo_update
from irl.cfg.schema import PPOConfig
from irl.models.networks import PolicyNetwork, ValueNetwork


def _flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters()])


def test_ppo_kl_penalty_changes_policy_update() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    base_policy = PolicyNetwork(obs_space, act_space)
    base_value = ValueNetwork(obs_space)

    init_pol = {k: v.clone() for k, v in base_policy.state_dict().items()}
    init_val = {k: v.clone() for k, v in base_value.state_dict().items()}

    N = 32
    obs = rng.standard_normal((N, 4)).astype(np.float32)

    with torch.no_grad():
        dist0 = base_policy.distribution(torch.as_tensor(obs, dtype=torch.float32))
        actions_t = dist0.sample()
        old_logp_t = dist0.log_prob(actions_t)

    actions = actions_t.detach().cpu().numpy()
    old_logp = old_logp_t.detach().cpu().numpy().astype(np.float32)

    advantages = rng.standard_normal(N).astype(np.float32)
    value_targets = rng.standard_normal(N).astype(np.float32)

    batch = {"obs": obs, "actions": actions, "old_log_probs": old_logp}

    cfg = PPOConfig(
        steps_per_update=N,
        minibatches=1,
        epochs=2,
        learning_rate=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.0,
        value_coef=0.5,
        value_clip_range=0.0,
        kl_penalty_coef=0.0,
        kl_stop=0.0,
    )

    policy_a = PolicyNetwork(obs_space, act_space)
    value_a = ValueNetwork(obs_space)
    policy_a.load_state_dict(init_pol)
    value_a.load_state_dict(init_val)

    torch.manual_seed(123)
    _ = ppo_update(
        policy_a,
        value_a,
        batch,
        advantages,
        value_targets,
        cfg,
        return_stats=False,
    )
    pa = _flat_params(policy_a)

    policy_b = PolicyNetwork(obs_space, act_space)
    value_b = ValueNetwork(obs_space)
    policy_b.load_state_dict(init_pol)
    value_b.load_state_dict(init_val)

    torch.manual_seed(123)
    _ = ppo_update(
        policy_b,
        value_b,
        batch,
        advantages,
        value_targets,
        replace(cfg, kl_penalty_coef=10.0),
        return_stats=False,
    )
    pb = _flat_params(policy_b)

    max_abs = float((pa - pb).abs().max().item())
    assert max_abs > 1e-7
