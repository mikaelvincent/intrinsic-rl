from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from irl.cfg import Config, validate_config
from irl.trainer import train as run_train


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


try:
    register(id="NeverTerminating-v0", entry_point=_NeverTerminatingEnv)
except Exception:
    pass


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_episode_metrics_nan_when_no_episodes(tmp_path: Path) -> None:
    base = Config()
    cfg = replace(
        base,
        device="cpu",
        method="vanilla",
        env=replace(
            base.env,
            id="NeverTerminating-v0",
            vec_envs=1,
            frame_skip=1,
            async_vector=False,
        ),
        ppo=replace(base.ppo, steps_per_update=1, minibatches=1, epochs=1),
        logging=replace(base.logging, csv_interval=1, checkpoint_interval=100_000),
        evaluation=replace(base.evaluation, interval_steps=0, episodes=1),
        adaptation=replace(base.adaptation, enabled=False),
    )
    validate_config(cfg)

    out_dir = run_train(cfg, total_steps=1, run_dir=tmp_path / "run", resume=False)

    rows = _read_rows(out_dir / "logs" / "scalars.csv")
    assert rows
    r = rows[-1]

    assert int(float(r["episode_count"])) == 0
    assert math.isnan(float(r["episode_return_mean"]))
    assert math.isnan(float(r["episode_length_mean"]))
