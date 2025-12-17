from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from irl.cfg import Config, validate_config
from irl.trainer import train as run_train


class _TimeoutMaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self._init_seed = 0 if seed is None else int(seed)
        self._episode = 0
        self._t = 0
        self._mode = "term"

    def reset(self, *, seed=None, options=None):
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


try:
    register(id="TimeoutMask-v0", entry_point=_TimeoutMaskEnv)
except Exception:
    pass


def _make_cfg() -> Config:
    base = Config()

    env_cfg = replace(
        base.env,
        id="TimeoutMask-v0",
        vec_envs=2,
        frame_skip=1,
        domain_randomization=False,
        discrete_actions=True,
        async_vector=False,
    )
    ppo_cfg = replace(
        base.ppo,
        steps_per_update=1,
        minibatches=1,
        epochs=1,
        entropy_coef=0.0,
    )
    intrinsic_cfg = replace(base.intrinsic, eta=0.1)
    log_cfg = replace(base.logging, csv_interval=1, checkpoint_interval=100_000)
    eval_cfg = replace(base.evaluation, interval_steps=100_000, episodes=1)
    adapt_cfg = replace(base.adaptation, enabled=False)

    cfg = replace(
        base,
        device="cpu",
        method="ride",
        env=env_cfg,
        ppo=ppo_cfg,
        intrinsic=intrinsic_cfg,
        logging=log_cfg,
        evaluation=eval_cfg,
        adaptation=adapt_cfg,
    )
    validate_config(cfg)
    return cfg


def test_intrinsic_not_masked_on_truncations(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_timeout_mask"
    cfg = _make_cfg()
    out = run_train(cfg, total_steps=4, run_dir=run_dir, resume=False)

    csv_path = out / "logs" / "scalars.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_step: dict[int, dict[str, str]] = {}
    for r in rows:
        try:
            by_step[int(r["step"])] = r
        except Exception:
            continue

    r2 = float(by_step[2]["r_int_mean"])
    r4 = float(by_step[4]["r_int_mean"])

    assert np.isfinite(r2) and r2 > 0.0
    assert abs(r4) < 1e-12
