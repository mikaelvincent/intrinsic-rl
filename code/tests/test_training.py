from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.envs.registration import register
from torch import nn

from irl.cfg import Config, validate_config
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.ride import RIDE
from irl.trainer import train as run_train
from irl.trainer.rollout import collect_rollout
from irl.utils.checkpoint import load_checkpoint
from irl.utils.loggers import get_logger


class _TimeoutMaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._init_seed = 0 if seed is None else int(seed)
        self._episode = 0
        self._t = 0
        self._mode = "term"

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._init_seed = int(seed)
        self._episode += 1
        self._t = 0
        self._mode = "term" if self._episode > 1 else ("term" if (self._init_seed % 2 == 1) else "trunc")
        obs = np.array([float(self._episode), 0.0, 0.0, 0.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = np.array([float(self._episode), float(self._t), float(action), 1.0], dtype=np.float32)
        terminated = self._t >= 1 and self._mode == "term"
        truncated = self._t >= 1 and self._mode == "trunc"
        return obs, 0.0, bool(terminated), bool(truncated), {}

    def close(self) -> None:
        return


class _ObsNormCountEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        _ = action
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
        return obs, 0.0, False, False, {}

    def close(self) -> None:
        return


class _NeverTerminatingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        _ = action
        return np.zeros((4,), dtype=np.float32), 0.0, False, False, {}

    def close(self) -> None:
        return


for _id, _cls in (
    ("TimeoutMask-v0", _TimeoutMaskEnv),
    ("ObsNormCount-v0", _ObsNormCountEnv),
    ("NeverTerminating-v0", _NeverTerminatingEnv),
):
    try:
        register(id=_id, entry_point=_cls)
    except Exception:
        pass


def _make_cfg(
    *,
    env_id: str,
    method: str,
    vec_envs: int,
    rollout_steps_per_env: int,
    minibatches: int,
    epochs: int,
    eta: float,
) -> Config:
    base = Config()
    env_cfg = replace(
        base.env,
        id=str(env_id),
        vec_envs=int(vec_envs),
        frame_skip=1,
        domain_randomization=False,
        discrete_actions=True,
        async_vector=False,
    )
    ppo_cfg = replace(
        base.ppo,
        rollout_steps_per_env=int(rollout_steps_per_env),
        minibatches=int(minibatches),
        epochs=int(epochs),
        entropy_coef=0.0,
    )
    intrinsic_cfg = replace(base.intrinsic, eta=float(eta))
    log_cfg = replace(base.logging, csv_interval=1, checkpoint_interval=100_000)
    eval_cfg = replace(base.evaluation, interval_steps=100_000, episodes=1)
    cfg = replace(
        base,
        device="cpu",
        method=str(method),
        env=env_cfg,
        ppo=ppo_cfg,
        intrinsic=intrinsic_cfg,
        logging=log_cfg,
        evaluation=eval_cfg,
        exp=replace(base.exp, deterministic=True),
    )
    validate_config(cfg)
    return cfg


def test_train_refuses_dirty_run_dir_without_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dirty"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "ckpt_latest.pt").write_bytes(b"not a real checkpoint")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "scalars.csv").write_text("step,reward_total_mean\n0,0.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="resume"):
        run_train(Config(), total_steps=1, run_dir=run_dir, resume=False)


def test_intrinsic_not_masked_on_truncations(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="TimeoutMask-v0",
        method="ride",
        vec_envs=2,
        rollout_steps_per_env=1,
        minibatches=1,
        epochs=1,
        eta=0.1,
    )
    out_dir = run_train(cfg, total_steps=4, run_dir=tmp_path / "run_timeout", resume=False)

    csv_path = out_dir / "logs" / "scalars.csv"
    rows: dict[int, dict[str, str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows[int(r["step"])] = r
            except Exception:
                continue

    r2 = float(rows[2]["r_int_mean"])
    r4 = float(rows[4]["r_int_mean"])
    assert np.isfinite(r2) and r2 > 0.0
    assert abs(r4) < 1e-12


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
        self.observation_space = gym.spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        self._t = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._t += 1
        obs = np.array([float(self._t)], dtype=np.float32)
        truncated = self._t >= 2
        if truncated:
            self._t = 0
            obs = np.array([0.0], dtype=np.float32)
        return obs, 0.0, False, bool(truncated), {}

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


def test_ride_binning_reset_applies_on_episode_start_in_rollout() -> None:
    env = _AutoResetTruncEnv()
    try:
        device = torch.device("cpu")
        obs0, _ = env.reset(seed=0)

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


def _read_scalars_by_step(path: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["step"])] = dict(r)
            except Exception:
                continue
    return out


def test_glpe_intrinsic_taper_weight_reaches_zero(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="ObsNormCount-v0",
        method="glpe",
        vec_envs=1,
        rollout_steps_per_env=2,
        minibatches=1,
        epochs=1,
        eta=0.1,
    )
    cfg = replace(cfg, intrinsic=replace(cfg.intrinsic, taper_start_frac=0.0, taper_end_frac=0.5))

    out_dir = run_train(cfg, total_steps=8, run_dir=tmp_path / "run_glpe_taper", resume=False)

    rows = _read_scalars_by_step(out_dir / "logs" / "scalars.csv")
    w2 = float(rows[2]["intrinsic_taper_weight"])
    w4 = float(rows[4]["intrinsic_taper_weight"])
    w6 = float(rows[6]["intrinsic_taper_weight"])

    assert w2 > w4 > w6
    assert abs(w6) < 1e-12
    assert abs(float(rows[6]["r_int_mean"])) < 1e-12
    assert abs(float(rows[6]["intrinsic_eta_effective"])) < 1e-12


def test_episode_metrics_nan_when_no_episodes(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="NeverTerminating-v0",
        method="vanilla",
        vec_envs=1,
        rollout_steps_per_env=1,
        minibatches=1,
        epochs=1,
        eta=0.0,
    )
    out_dir = run_train(cfg, total_steps=1, run_dir=tmp_path / "run_no_episodes", resume=False)

    rows = _read_scalars_by_step(out_dir / "logs" / "scalars.csv")
    r = rows[max(rows.keys())]
    assert int(float(r["episode_count"])) == 0
    assert math.isnan(float(r["episode_return_mean"]))
    assert math.isnan(float(r["episode_length_mean"]))


def test_run_meta_written_and_embedded(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="MountainCar-v0",
        method="vanilla",
        vec_envs=1,
        rollout_steps_per_env=1,
        minibatches=1,
        epochs=1,
        eta=0.0,
    )
    out_dir = run_train(cfg, total_steps=1, run_dir=tmp_path / "run_meta", resume=False)

    meta_path = out_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    payload = load_checkpoint(out_dir / "checkpoints" / "ckpt_latest.pt", map_location="cpu")
    assert payload.get("run_meta") == meta
