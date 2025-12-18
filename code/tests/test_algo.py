from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg.schema import PPOConfig
from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.pipelines.policy_rollout import iter_policy_rollout


class _ValueSequence(nn.Module):
    def __init__(self, v_t, v_tp1):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))
        self.v_t = torch.as_tensor(v_t, dtype=torch.float32).view(-1)
        self.v_tp1 = torch.as_tensor(v_tp1, dtype=torch.float32).view(-1)
        self._calls = 0

    def forward(self, _obs):
        self._calls += 1
        return self.v_t if self._calls == 1 else self.v_tp1


def test_compute_gae_bootstraps_timeouts_and_dones() -> None:
    obs = torch.zeros((3, 1, 1), dtype=torch.float32)
    next_obs = torch.zeros((3, 1, 1), dtype=torch.float32)

    rewards = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    v_t = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    v_tp1 = torch.tensor([20.0, 30.0, 40.0], dtype=torch.float32)

    dones = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)
    vf_term = _ValueSequence(v_t, v_tp1)
    adv_term, vt_term = compute_gae(
        {"obs": obs, "next_observations": next_obs, "rewards": rewards, "dones": dones},
        value_fn=vf_term,
        gamma=1.0,
        lam=1.0,
        bootstrap_on_timeouts=False,
    )
    assert torch.allclose(
        adv_term, torch.tensor([-7.0, -18.0, -29.0], dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(vt_term, torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32), atol=1e-6)

    terminals = torch.tensor([[0.0], [0.0], [0.0]], dtype=torch.float32)
    truncs = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)
    vf_boot = _ValueSequence(v_t, v_tp1)
    adv_boot, vt_boot = compute_gae(
        {
            "obs": obs,
            "next_observations": next_obs,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncs,
        },
        value_fn=vf_boot,
        gamma=1.0,
        lam=1.0,
        bootstrap_on_timeouts=True,
    )
    assert torch.allclose(
        adv_boot, torch.tensor([33.0, 22.0, 11.0], dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(
        vt_boot, torch.tensor([43.0, 42.0, 41.0], dtype=torch.float32), atol=1e-6
    )


def test_compute_gae_does_not_leak_across_truncation() -> None:
    obs = torch.zeros((4, 1, 1), dtype=torch.float32)
    next_obs = torch.zeros((4, 1, 1), dtype=torch.float32)

    rewards = torch.tensor([[0.0], [0.0], [10.0], [0.0]], dtype=torch.float32)
    terminals = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32)
    truncations = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=torch.float32)

    vf = _ValueSequence(
        v_t=[0.0, 0.0, 0.0, 0.0],
        v_tp1=[0.0, 5.0, 0.0, 0.0],
    )
    adv, v_targets = compute_gae(
        {
            "obs": obs,
            "next_observations": next_obs,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncations,
        },
        value_fn=vf,
        gamma=1.0,
        lam=1.0,
        bootstrap_on_timeouts=True,
    )
    expected = torch.tensor([5.0, 5.0, 10.0, 0.0], dtype=torch.float32)
    assert torch.allclose(adv, expected, atol=1e-6)
    assert torch.allclose(v_targets, expected, atol=1e-6)


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters()])


def test_ppo_update_kl_penalty_changes_policy() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    base_policy = PolicyNetwork(obs_space, act_space)
    base_value = ValueNetwork(obs_space)

    init_pol = {k: v.clone() for k, v in base_policy.state_dict().items()}
    init_val = {k: v.clone() for k, v in base_value.state_dict().items()}

    N = 64
    obs = rng.standard_normal((N, 4)).astype(np.float32)

    with torch.no_grad():
        dist0 = base_policy.distribution(torch.as_tensor(obs, dtype=torch.float32))
        actions_t = dist0.sample()
        old_logp_t = dist0.log_prob(actions_t)

    batch = {
        "obs": obs,
        "actions": actions_t.detach().cpu().numpy(),
        "old_log_probs": old_logp_t.detach().cpu().numpy().astype(np.float32),
    }
    advantages = rng.standard_normal(N).astype(np.float32)
    value_targets = rng.standard_normal(N).astype(np.float32)

    cfg = PPOConfig(
        steps_per_update=N,
        minibatches=2,
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
    stats = ppo_update(
        policy_a,
        value_a,
        batch,
        advantages,
        value_targets,
        cfg,
        return_stats=True,
    )
    assert stats is not None
    assert np.isfinite(float(stats["approx_kl"]))

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
        PPOConfig(**{**cfg.__dict__, "kl_penalty_coef": 10.0}),
        return_stats=False,
    )

    diff = float((_flat_params(policy_a) - _flat_params(policy_b)).abs().max().item())
    assert diff > 1e-7


class _BoundedBoxEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        self._t = 0
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        lo = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        hi = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        assert a.shape == lo.shape == hi.shape
        assert np.all(a >= lo - 1e-6)
        assert np.all(a <= hi + 1e-6)

        self._t += 1
        terminated = self._t >= 3
        obs = np.zeros((4,), dtype=np.float32)
        return obs, 0.0, bool(terminated), False, {}

    def close(self) -> None:
        return


def test_policy_rollout_continuous_actions_within_bounds() -> None:
    env = _BoundedBoxEnv()
    try:
        policy = PolicyNetwork(env.observation_space, env.action_space).to(torch.device("cpu"))
        policy.eval()

        for mode in ("mode", "sample"):
            obs0, _ = env.reset(seed=0)
            steps = 0
            for _step in iter_policy_rollout(
                env=env,
                policy=policy,
                obs0=obs0,
                act_space=env.action_space,
                device=torch.device("cpu"),
                policy_mode=mode,
                normalize_obs=None,
                max_steps=5,
            ):
                steps += 1
            assert steps == 3
    finally:
        env.close()
