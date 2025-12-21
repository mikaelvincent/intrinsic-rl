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
from irl.trainer.rollout import collect_rollout
from irl.trainer.update_steps import compute_advantages
from irl.utils.loggers import get_logger


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


def test_compute_gae_handles_timeouts_and_layout() -> None:
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
    assert torch.allclose(adv_term, torch.tensor([-7.0, -18.0, -29.0], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(vt_term, torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32), atol=1e-6)

    terminals = torch.zeros((3, 1), dtype=torch.float32)
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
    assert torch.allclose(adv_boot, torch.tensor([33.0, 22.0, 11.0], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(vt_boot, torch.tensor([43.0, 42.0, 41.0], dtype=torch.float32), atol=1e-6)

    obs2 = torch.zeros((4, 1, 1), dtype=torch.float32)
    next_obs2 = torch.zeros((4, 1, 1), dtype=torch.float32)
    rewards2 = torch.tensor([[0.0], [0.0], [10.0], [0.0]], dtype=torch.float32)
    terminals2 = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32)
    truncations2 = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=torch.float32)
    vf2 = _ValueSequence(v_t=[0.0, 0.0, 0.0, 0.0], v_tp1=[0.0, 5.0, 0.0, 0.0])
    adv2, vt2 = compute_gae(
        {
            "obs": obs2,
            "next_observations": next_obs2,
            "rewards": rewards2,
            "terminals": terminals2,
            "truncations": truncations2,
        },
        value_fn=vf2,
        gamma=1.0,
        lam=1.0,
        bootstrap_on_timeouts=True,
    )
    expected = torch.tensor([5.0, 5.0, 10.0, 0.0], dtype=torch.float32)
    assert torch.allclose(adv2, expected, atol=1e-6)
    assert torch.allclose(vt2, expected, atol=1e-6)

    T, B = 3, 2
    obs_tb = torch.zeros((T, B, 1), dtype=torch.float32)
    next_obs_tb = torch.zeros((T, B, 1), dtype=torch.float32)
    obs_bt = obs_tb.transpose(0, 1)
    next_obs_bt = next_obs_tb.transpose(0, 1)

    rewards3 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    dones3 = torch.zeros((T, B), dtype=torch.float32)
    dones3[-1] = 1.0

    vf_time_major = _ValueSequence(
        v_t=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        v_tp1=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    adv_tb, vt_tb = compute_gae(
        {"obs": obs_tb, "next_observations": next_obs_tb, "rewards": rewards3, "dones": dones3},
        value_fn=vf_time_major,
        gamma=0.99,
        lam=0.95,
        bootstrap_on_timeouts=False,
    )

    vf_batch_major = _ValueSequence(
        v_t=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        v_tp1=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    adv_bt, vt_bt = compute_gae(
        {"obs": obs_bt, "next_observations": next_obs_bt, "rewards": rewards3, "dones": dones3},
        value_fn=vf_batch_major,
        gamma=0.99,
        lam=0.95,
        bootstrap_on_timeouts=False,
    )
    assert torch.allclose(adv_tb, adv_bt, atol=1e-6)
    assert torch.allclose(vt_tb, vt_bt, atol=1e-6)


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

    obs = rng.standard_normal((64, 4)).astype(np.float32)
    with torch.no_grad():
        dist0 = base_policy.distribution(torch.as_tensor(obs, dtype=torch.float32))
        actions_t = dist0.sample()
        old_logp_t = dist0.log_prob(actions_t)

    batch = {
        "obs": obs,
        "actions": actions_t.detach().cpu().numpy(),
        "old_log_probs": old_logp_t.detach().cpu().numpy().astype(np.float32),
    }
    advantages = rng.standard_normal(64).astype(np.float32)
    value_targets = rng.standard_normal(64).astype(np.float32)

    cfg = PPOConfig(
        steps_per_update=64,
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
    assert stats is not None and np.isfinite(float(stats["approx_kl"]))

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
        return np.zeros((4,), dtype=np.float32), 0.0, bool(terminated), False, {}

    def close(self) -> None:
        return


def test_policy_rollout_continuous_actions_within_bounds() -> None:
    env = _BoundedBoxEnv()
    try:
        policy = PolicyNetwork(env.observation_space, env.action_space).to(torch.device("cpu"))
        policy.eval()

        for mode in ("mode", "sample"):
            obs0, _ = env.reset(seed=0)
            steps = sum(
                1
                for _ in iter_policy_rollout(
                    env=env,
                    policy=policy,
                    obs0=obs0,
                    act_space=env.action_space,
                    device=torch.device("cpu"),
                    policy_mode=mode,
                    normalize_obs=None,
                    max_steps=5,
                )
            )
            assert steps == 3
    finally:
        env.close()


class _IdentityObsNorm:
    def update(self, _x: np.ndarray) -> None:
        return

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)


class _ZeroPolicy:
    def __init__(self, device: torch.device) -> None:
        self._device = device

    def act(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = int(obs_t.shape[0]) if obs_t.dim() >= 2 else 1
        a = torch.zeros((b,), dtype=torch.int64, device=self._device)
        logp = torch.zeros((b,), dtype=torch.float32, device=self._device)
        return a, logp


class _AutoResetNoFinalObsEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        self._t = 0
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._t += 1
        truncated = self._t >= 1
        terminated = False

        if truncated:
            self._t = 0
            obs = np.array([0.0], dtype=np.float32)
        else:
            obs = np.array([1.0], dtype=np.float32)

        return obs, 0.0, bool(terminated), bool(truncated), {}

    def close(self) -> None:
        return


class _ValueByObs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x0 = obs[..., 0].to(dtype=torch.float32)
        return torch.where(x0 < 0.5, torch.full_like(x0, 10.0), torch.zeros_like(x0))


def test_timeout_bootstrap_disabled_without_final_observation() -> None:
    device = torch.device("cpu")
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    env = _AutoResetNoFinalObsEnv()
    try:
        obs0, _ = env.reset(seed=0)

        rollout = collect_rollout(
            env=env,
            policy=_ZeroPolicy(device),
            actor_policy=None,
            obs=obs0,
            obs_space=obs_space,
            act_space=act_space,
            is_image=False,
            obs_norm=_IdentityObsNorm(),
            intrinsic_module=None,
            use_intrinsic=False,
            method_l="vanilla",
            T=1,
            B=1,
            device=device,
            logger=get_logger("test_timeout_bootstrap_guard"),
        )

        assert float(rollout.dones_seq.reshape(-1)[0]) == 1.0
        assert float(rollout.terminals_seq.reshape(-1)[0]) == 0.0
        assert float(rollout.truncations_seq.reshape(-1)[0]) == 1.0
        assert float(rollout.timeouts_no_final_obs_seq.reshape(-1)[0]) == 1.0

        adv_out = compute_advantages(
            rollout=rollout,
            rewards_total_seq=rollout.rewards_ext_seq,
            value_fn=_ValueByObs(),
            gamma=1.0,
            lam=1.0,
            device=device,
            profile_cuda_sync=False,
            maybe_cuda_sync=lambda _d, _e: None,
        )

        assert torch.allclose(adv_out.advantages, torch.zeros_like(adv_out.advantages), atol=1e-6)
        assert torch.allclose(adv_out.value_targets, torch.zeros_like(adv_out.value_targets), atol=1e-6)
    finally:
        env.close()


class _NoAutoResetDoneEnv:
    def __init__(self) -> None:
        self._done = False
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        self.reset_calls += 1
        self._done = False
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        if self._done:
            raise RuntimeError("step called after done")
        self.step_calls += 1
        self._done = True
        return np.array([0.0], dtype=np.float32), 0.0, True, False, {}

    def close(self) -> None:
        return


def test_collect_rollout_single_env_resets_on_done() -> None:
    device = torch.device("cpu")
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    env = _NoAutoResetDoneEnv()
    try:
        obs0, _ = env.reset(seed=0)

        rollout = collect_rollout(
            env=env,
            policy=_ZeroPolicy(device),
            actor_policy=None,
            obs=obs0,
            obs_space=obs_space,
            act_space=act_space,
            is_image=False,
            obs_norm=_IdentityObsNorm(),
            intrinsic_module=None,
            use_intrinsic=False,
            method_l="vanilla",
            T=3,
            B=1,
            device=device,
            logger=get_logger("test_single_env_done_reset"),
        )

        assert int(env.reset_calls) == 4
        assert int(env.step_calls) == 3
        assert float(rollout.terminals_seq.sum()) == 3.0
        assert np.allclose(rollout.obs_seq.reshape(-1), np.ones((3,), dtype=np.float32))
        assert np.allclose(rollout.next_obs_seq.reshape(-1), np.zeros((3,), dtype=np.float32))

        final_obs = np.asarray(rollout.final_env_obs, dtype=np.float32).reshape(-1)
        assert np.allclose(final_obs, np.array([1.0], dtype=np.float32))
    finally:
        env.close()


class _NeverDoneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        _ = action
        return np.zeros((4,), dtype=np.float32), 0.0, False, False, {}

    def close(self) -> None:
        return


def test_ppo_uses_old_log_probs_from_rollout() -> None:
    device = torch.device("cpu")
    env = _NeverDoneEnv()
    try:
        obs0, _ = env.reset(seed=0)
        policy = PolicyNetwork(env.observation_space, env.action_space).to(device)
        value = ValueNetwork(env.observation_space).to(device)
        policy.eval()
        value.eval()

        T = 4
        rollout = collect_rollout(
            env=env,
            policy=policy,
            actor_policy=None,
            obs=obs0,
            obs_space=env.observation_space,
            act_space=env.action_space,
            is_image=False,
            obs_norm=_IdentityObsNorm(),
            intrinsic_module=None,
            use_intrinsic=False,
            method_l="vanilla",
            T=T,
            B=1,
            device=device,
            logger=get_logger("test_old_log_probs_rollout"),
        )

        N = int(rollout.T) * int(rollout.B)
        obs_flat = rollout.obs_seq.reshape(N, -1)
        acts_flat = rollout.actions_seq.reshape(N)

        cfg = PPOConfig(
            steps_per_update=int(N),
            minibatches=2,
            epochs=1,
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

        batch = {
            "obs": obs_flat,
            "actions": acts_flat,
            "old_log_probs": rollout.old_log_probs_seq.reshape(N).astype(np.float32),
        }

        advantages = np.zeros((N,), dtype=np.float32)
        value_targets = np.zeros((N,), dtype=np.float32)

        calls = 0
        orig = policy.distribution

        def _wrapped(obs, orig=orig):
            nonlocal calls
            calls += 1
            return orig(obs)

        policy.distribution = _wrapped

        _ = ppo_update(
            policy,
            value,
            batch,
            advantages,
            value_targets,
            cfg,
            return_stats=False,
        )

        assert int(calls) == int(cfg.minibatches) * int(cfg.epochs)
    finally:
        env.close()
