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


try:
    register(id="ObsNormCount-v0", entry_point=_ObsNormCountEnv)
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

    with pytest.raises(RuntimeError, match=r"resume"):
        run_train(Config(), total_steps=1, run_dir=run_dir, resume=False)


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
