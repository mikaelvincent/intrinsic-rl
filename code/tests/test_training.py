from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import register

from irl.cfg import Config, validate_config
from irl.trainer import train as run_train
from irl.utils.checkpoint import load_checkpoint


class _StepBudgetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        return np.zeros((2,), dtype=np.float32), {}

    def step(self, action):
        _ = action
        obs = np.zeros((2,), dtype=np.float32)
        return obs, 0.0, False, False, {}

    def close(self) -> None:
        return


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

        if self._episode == 1:
            self._mode = "term" if (int(self._init_seed) % 2 == 1) else "trunc"
        else:
            self._mode = "term"

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


class _DummyImageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, h: int = 32, w: int = 32, seed: int | None = None) -> None:
        super().__init__()
        self.H, self.W = int(h), int(w)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(3)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.integers(0, 256, size=(self.H, self.W, 3), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        _ = action
        self._t += 1
        obs = self._rng.integers(0, 256, size=(self.H, self.W, 3), dtype=np.uint8)
        reward = 0.1
        terminated = self._t >= 5
        return obs, reward, bool(terminated), False, {}

    def close(self) -> None:
        return


for _id, _cls in (
    ("StepBudget-v0", _StepBudgetEnv),
    ("TimeoutMask-v0", _TimeoutMaskEnv),
    ("ObsNormCount-v0", _ObsNormCountEnv),
    ("DummyImage-v0", _DummyImageEnv),
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
    steps_per_update: int,
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
        steps_per_update=int(steps_per_update),
        minibatches=int(minibatches),
        epochs=int(epochs),
        entropy_coef=0.0,
    )
    intrinsic_cfg = replace(base.intrinsic, eta=float(eta))
    log_cfg = replace(base.logging, csv_interval=1, checkpoint_interval=100_000)
    eval_cfg = replace(base.evaluation, interval_steps=100_000, episodes=1)
    adapt_cfg = replace(base.adaptation, enabled=False)
    cfg = replace(
        base,
        device="cpu",
        method=str(method),
        env=env_cfg,
        ppo=ppo_cfg,
        intrinsic=intrinsic_cfg,
        logging=log_cfg,
        evaluation=eval_cfg,
        adaptation=adapt_cfg,
    )
    validate_config(cfg)
    return cfg


def _read_csv_column(path: Path, col: str) -> list[float]:
    out: list[float] = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or col not in set(r.fieldnames):
            return []
        for row in r:
            try:
                out.append(float(row[col]))
            except Exception:
                continue
    return out


def test_train_refuses_dirty_run_dir_without_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dirty"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "ckpt_latest.pt").write_bytes(b"not a real checkpoint")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "scalars.csv").write_text("step,reward_total_mean\n0,0.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="resume"):
        run_train(Config(), total_steps=1, run_dir=run_dir, resume=False)


def test_total_steps_aligns_to_vec_envs_budget(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="StepBudget-v0",
        method="vanilla",
        vec_envs=2,
        steps_per_update=2,
        minibatches=1,
        epochs=1,
        eta=0.0,
    )
    out_dir = run_train(cfg, total_steps=5, run_dir=tmp_path / "run_budget", resume=False)

    payload = load_checkpoint(out_dir / "checkpoints" / "ckpt_latest.pt", map_location="cpu")
    step = int(payload.get("step", -1))

    assert step == 4
    assert step <= 5
    assert step % int(cfg.env.vec_envs) == 0


def test_obs_norm_counts_once_per_transition(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="ObsNormCount-v0",
        method="vanilla",
        vec_envs=2,
        steps_per_update=2,
        minibatches=1,
        epochs=1,
        eta=0.0,
    )
    out_dir = run_train(cfg, total_steps=4, run_dir=tmp_path / "run_obs_norm_count", resume=False)

    payload = load_checkpoint(out_dir / "checkpoints" / "ckpt_latest.pt", map_location="cpu")
    step = int(payload.get("step", -1))
    assert step == 4

    obs_norm = payload.get("obs_norm")
    assert isinstance(obs_norm, dict)

    count = float(obs_norm.get("count", float("nan")))
    assert np.isfinite(count)
    assert abs(count - float(step)) < 1e-6


def test_intrinsic_not_masked_on_truncations(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="TimeoutMask-v0",
        method="ride",
        vec_envs=2,
        steps_per_update=1,
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


def test_trainer_image_pipeline_riac_logs_intrinsic(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="DummyImage-v0",
        method="riac",
        vec_envs=1,
        steps_per_update=6,
        minibatches=1,
        epochs=1,
        eta=0.1,
    )
    out_dir = run_train(cfg, total_steps=6, run_dir=tmp_path / "run_riac_img", resume=False)

    csv_path = out_dir / "logs" / "scalars.csv"
    vals = _read_csv_column(csv_path, "r_int_mean")
    assert vals and any(np.isfinite(v) for v in vals)
