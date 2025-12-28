from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg.schema import PPOConfig
from irl.intrinsic import RunningRMS
from irl.intrinsic.icm import ICM
from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.pipelines.policy_rollout import iter_policy_rollout
from irl.trainer.rollout import collect_rollout
from irl.trainer.runtime_utils import _apply_final_observation
from irl.trainer.update_steps import compute_advantages, compute_intrinsic_rewards
from irl.utils.loggers import get_logger


class _ValueSequence(nn.Module):
    def __init__(self, v_t: object, v_tp1: object) -> None:
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))
        self._v_t = torch.as_tensor(v_t, dtype=torch.float32).view(-1)
        self._v_tp1 = torch.as_tensor(v_tp1, dtype=torch.float32).view(-1)
        self._calls = 0

    def forward(self, _obs: torch.Tensor) -> torch.Tensor:
        self._calls += 1
        return self._v_t if self._calls == 1 else self._v_tp1


def test_compute_gae_bootstrap_timeouts_and_layout() -> None:
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
    assert torch.allclose(
        adv_boot, torch.tensor([33.0, 22.0, 11.0], dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(vt_boot, torch.tensor([43.0, 42.0, 41.0], dtype=torch.float32), atol=1e-6)

    T, B = 3, 2
    obs_tb = torch.zeros((T, B, 1), dtype=torch.float32)
    next_obs_tb = torch.zeros((T, B, 1), dtype=torch.float32)
    obs_bt = obs_tb.transpose(0, 1)
    next_obs_bt = next_obs_tb.transpose(0, 1)

    rewards3 = torch.arange(1, T * B + 1, dtype=torch.float32).reshape(T, B)
    dones3 = torch.zeros((T, B), dtype=torch.float32)
    dones3[-1] = 1.0

    vf_a = _ValueSequence(v_t=torch.zeros(T * B), v_tp1=torch.ones(T * B))
    adv_a, vt_a = compute_gae(
        {"obs": obs_tb, "next_observations": next_obs_tb, "rewards": rewards3, "dones": dones3},
        value_fn=vf_a,
        gamma=0.99,
        lam=0.95,
        bootstrap_on_timeouts=False,
    )

    vf_b = _ValueSequence(v_t=torch.zeros(T * B), v_tp1=torch.ones(T * B))
    adv_b, vt_b = compute_gae(
        {"obs": obs_bt, "next_observations": next_obs_bt, "rewards": rewards3, "dones": dones3},
        value_fn=vf_b,
        gamma=0.99,
        lam=0.95,
        bootstrap_on_timeouts=False,
    )
    assert torch.allclose(adv_a, adv_b, atol=1e-6)
    assert torch.allclose(vt_a, vt_b, atol=1e-6)


def test_ppo_update_uses_provided_old_log_probs() -> None:
    rng = np.random.default_rng(0)
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    obs = rng.standard_normal((64, 4)).astype(np.float32)
    actions = rng.integers(0, int(act_space.n), size=(64,), endpoint=False, dtype=np.int64)
    old_log_probs = np.full((64,), np.nan, dtype=np.float32)

    batch = {"obs": obs, "actions": actions, "old_log_probs": old_log_probs}
    advantages = rng.standard_normal(64).astype(np.float32)
    value_targets = rng.standard_normal(64).astype(np.float32)

    cfg = PPOConfig(rollout_steps_per_env=64, minibatches=2, epochs=1)
    with pytest.raises(ValueError, match=r"old_log_probs"):
        _ = ppo_update(
            policy,
            value,
            batch,
            advantages,
            value_targets,
            cfg,
            return_stats=False,
        )


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


def test_apply_final_observation_vector_and_scalar() -> None:
    next_obs = np.array([[0.0], [1.0]], dtype=np.float32)
    done = np.array([True, False], dtype=bool)
    infos = {
        "final_observation": np.array([np.array([42.0], dtype=np.float32), None], dtype=object),
    }
    fixed = _apply_final_observation(next_obs, done, infos)
    assert np.allclose(fixed, np.array([[42.0], [1.0]], dtype=np.float32))

    next_obs1 = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    done1 = np.array([True], dtype=bool)
    infos1 = {"final_observation": np.array([10.0, 11.0, 12.0], dtype=np.float32)}
    fixed1 = _apply_final_observation(next_obs1, done1, infos1)
    assert np.allclose(fixed1, np.array([10.0, 11.0, 12.0], dtype=np.float32))


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


def test_timeouts_without_final_observation_mask_gae_and_intrinsic() -> None:
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
            logger=get_logger("test_timeout_mask"),
        )
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
        assert torch.allclose(
            adv_out.value_targets, torch.zeros_like(adv_out.value_targets), atol=1e-6
        )

        icm = ICM(obs_space, act_space, device="cpu")
        before_sd = {k: v.detach().clone() for k, v in icm.state_dict().items()}
        before_opt = len(getattr(icm, "_opt").state)

        out = compute_intrinsic_rewards(
            rollout=rollout,
            intrinsic_module=icm,
            use_intrinsic=True,
            method_l="icm",
            eta=0.1,
            r_clip=5.0,
            int_rms=RunningRMS(beta=0.99, eps=1e-8),
            device=device,
            profile_cuda_sync=False,
            maybe_cuda_sync=lambda _d, _e: None,
            total_steps=10,
            global_step_start=0,
            taper_start_frac=None,
            taper_end_frac=None,
        )

        assert int(out.module_metrics.get("invalid_timeouts_no_final_obs_count", 0.0)) == 1
        assert np.allclose(out.rewards_total_seq, rollout.rewards_ext_seq)

        if out.r_int_raw_flat is not None:
            assert np.allclose(out.r_int_raw_flat, np.zeros((1,), dtype=np.float32))
        if out.r_int_scaled_flat is not None:
            assert np.allclose(out.r_int_scaled_flat, np.zeros((1,), dtype=np.float32))

        after_sd = icm.state_dict()
        for k, v in before_sd.items():
            assert torch.equal(after_sd[k], v)
        assert len(getattr(icm, "_opt").state) == before_opt
    finally:
        env.close()
