from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from irl.trainer.rollout import collect_rollout
from irl.trainer.update_steps import compute_advantages
from irl.utils.loggers import get_logger


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
            obs = np.array([0.0], dtype=np.float32)  # post-reset obs
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
        assert torch.allclose(
            adv_out.value_targets, torch.zeros_like(adv_out.value_targets), atol=1e-6
        )
    finally:
        env.close()
