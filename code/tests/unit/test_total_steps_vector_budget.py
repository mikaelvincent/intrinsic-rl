from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import register

from irl.cfg import Config, validate_config
from irl.experiments import run_training_suite
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


try:
    register(id="StepBudget-v0", entry_point=_StepBudgetEnv)
except Exception:
    pass


def _make_cfg(*, vec_envs: int) -> Config:
    base = Config()
    env_cfg = replace(
        base.env,
        id="StepBudget-v0",
        vec_envs=int(vec_envs),
        frame_skip=1,
        domain_randomization=False,
        discrete_actions=True,
        async_vector=False,
    )
    ppo_cfg = replace(
        base.ppo,
        steps_per_update=2,
        minibatches=1,
        epochs=1,
        entropy_coef=0.0,
    )
    intrinsic_cfg = replace(base.intrinsic, eta=0.0)
    log_cfg = replace(base.logging, csv_interval=1, checkpoint_interval=100_000)
    eval_cfg = replace(base.evaluation, interval_steps=100_000, episodes=1)
    adapt_cfg = replace(base.adaptation, enabled=False)

    cfg = replace(
        base,
        device="cpu",
        method="vanilla",
        env=env_cfg,
        ppo=ppo_cfg,
        intrinsic=intrinsic_cfg,
        logging=log_cfg,
        evaluation=eval_cfg,
        adaptation=adapt_cfg,
    )
    validate_config(cfg)
    return cfg


def test_total_steps_does_not_overshoot_with_vec_envs(tmp_path: Path) -> None:
    cfg = _make_cfg(vec_envs=2)
    out_dir = run_train(cfg, total_steps=5, run_dir=tmp_path / "run_budget", resume=False)

    ckpt_path = out_dir / "checkpoints" / "ckpt_latest.pt"
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    step = int(payload.get("step", -1))

    assert step == 4
    assert step <= 5
    assert step % int(cfg.env.vec_envs) == 0


def test_suite_aligns_total_steps_to_vec_envs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = configs_dir / "budget.yaml"
    cfg_path.write_text(
        """
seed: 1
device: "cpu"
method: "vanilla"
env:
  id: "StepBudget-v0"
  vec_envs: 2
ppo:
  steps_per_update: 2
  minibatches: 1
  epochs: 1
logging:
  csv_interval: 1
  checkpoint_interval: 100000
exp:
  total_steps: 5
""".lstrip(),
        encoding="utf-8",
    )

    import irl.experiments.training as training_module

    captured: list[int] = []

    def fake_run_train(cfg, *, total_steps: int, run_dir: Path, resume: bool):
        _ = cfg, run_dir, resume
        captured.append(int(total_steps))

    monkeypatch.setattr(training_module, "run_train", fake_run_train)

    run_training_suite(
        configs_dir=configs_dir,
        include=[],
        exclude=[],
        total_steps=999,
        runs_root=tmp_path / "runs_suite",
        seeds=[1],
        device="cpu",
        resume=False,
        auto_async=False,
    )

    assert captured == [4]
