from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from irl.cfg import Config, validate_config
from irl.trainer import train as run_train


class _DummyGateEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        reward = 0.0
        terminated = self._t >= 3
        truncated = False
        return obs, reward, terminated, truncated, {}


try:
    register(id="DummyGate-v0", entry_point=_DummyGateEnv)
except Exception:
    pass


def _make_cfg(*, method: str, eta: float = 0.1) -> Config:
    base = Config()

    env_cfg = replace(
        base.env,
        id="DummyGate-v0",
        vec_envs=1,
        frame_skip=1,
        domain_randomization=False,
        discrete_actions=True,
    )
    ppo_cfg = replace(
        base.ppo,
        steps_per_update=4,
        minibatches=1,
        epochs=1,
        entropy_coef=0.0,
    )
    intrinsic_cfg = replace(
        base.intrinsic,
        eta=float(eta),
        alpha_impact=0.0,
        alpha_lp=0.5,
    )
    log_cfg = replace(
        base.logging,
        csv_interval=1,
        checkpoint_interval=100_000,
    )
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


def test_glpe_lp_only_logs_gate_rate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_glpe_lp_only"
    cfg = _make_cfg(method="glpe_lp_only", eta=0.1)
    out = run_train(cfg, total_steps=8, run_dir=run_dir, resume=False)

    csv_path = out / "logs" / "scalars.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        cols = set(reader.fieldnames)

    assert "gate_rate" in cols
    assert "gate_rate_pct" in cols
