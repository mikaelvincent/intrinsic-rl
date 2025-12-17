from __future__ import annotations

import numpy as np
import torch
from torch import nn
import gymnasium as gym

from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.ride import RIDE
from irl.trainer.rollout import collect_rollout
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


class _AutoResetTruncEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=100.0, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._t += 1

        obs = np.array([float(self._t)], dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, object] = {}

        if self._t >= 2:
            truncated = True
            info["final_observation"] = obs.copy()
            self._t = 0
            obs = np.array([0.0], dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        return


def _make_ride(obs_space: gym.Space, act_space: gym.Space) -> RIDE:
    ride = RIDE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=1, hidden=(1, 1)),
        bin_size=10.0,
        alpha_impact=1.0,
    )

    with torch.no_grad():
        for m in ride.icm.encoder.modules():
            if isinstance(m, nn.Linear):
                m.weight.fill_(1.0)
                m.bias.fill_(0.0)

    return ride


def test_ride_binning_reset_applies_on_episode_start_in_rollout():
    env = _AutoResetTruncEnv()
    try:
        device = torch.device("cpu")
        obs0, _ = env.reset()

        ride = _make_ride(env.observation_space, env.action_space)
        policy = _ZeroPolicy(device)
        obs_norm = _IdentityObsNorm()

        r1 = collect_rollout(
            env=env,
            policy=policy,
            actor_policy=None,
            obs=obs0,
            obs_space=env.observation_space,
            act_space=env.action_space,
            is_image=False,
            obs_norm=obs_norm,
            intrinsic_module=ride,
            use_intrinsic=True,
            method_l="ride",
            T=4,
            B=1,
            device=device,
            logger=get_logger("test_ride_rollout"),
        )
        assert r1.r_int_raw_seq is not None
        v = r1.r_int_raw_seq.reshape(-1)

        assert v[1] < v[0] - 1e-6
        assert v[2] > v[1] + 1e-6
        assert np.isclose(float(v[0]), float(v[2]), atol=1e-6)
        assert np.isclose(float(v[1]), float(v[3]), atol=1e-6)

        r2 = collect_rollout(
            env=env,
            policy=policy,
            actor_policy=None,
            obs=r1.final_env_obs,
            obs_space=env.observation_space,
            act_space=env.action_space,
            is_image=False,
            obs_norm=obs_norm,
            intrinsic_module=ride,
            use_intrinsic=True,
            method_l="ride",
            T=1,
            B=1,
            device=device,
            logger=get_logger("test_ride_rollout_2"),
        )
        assert r2.r_int_raw_seq is not None
        v2 = r2.r_int_raw_seq.reshape(-1)

        assert np.isclose(float(v2[0]), float(v[0]), atol=1e-6)
    finally:
        env.close()
