from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from irl.cfg import Config, validate_config
from irl.trainer import train as run_train


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
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.integers(0, 256, size=(self.H, self.W, 3), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = self._rng.integers(0, 256, size=(self.H, self.W, 3), dtype=np.uint8)
        reward = 0.1
        terminated = self._t >= 5
        truncated = False
        return obs, reward, terminated, truncated, {}


try:
    register(id="DummyImage-v0", entry_point=_DummyImageEnv)
except Exception:
    pass


def _make_cfg(
    *,
    method: str,
    total_steps_per_update: int = 8,
    minibatches: int = 2,
    epochs: int = 1,
    eta: float = 0.0,
) -> Config:
    base = Config()
    env_cfg = replace(
        base.env,
        id="DummyImage-v0",
        vec_envs=1,
        frame_skip=1,
        domain_randomization=False,
        discrete_actions=True,
    )
    ppo_cfg = replace(
        base.ppo,
        steps_per_update=total_steps_per_update,
        minibatches=minibatches,
        epochs=epochs,
        entropy_coef=0.01,
    )
    intrinsic_cfg = replace(base.intrinsic, eta=eta)
    log_cfg = replace(base.logging, tb=False, csv_interval=1, checkpoint_interval=10_000)
    eval_cfg = replace(base.evaluation, interval_steps=10_000, episodes=1)
    adapt_cfg = replace(base.adaptation, enabled=False)
    cfg = replace(
        base,
        device="cpu",
        method=method,
        env=env_cfg,
        ppo=ppo_cfg,
        intrinsic=intrinsic_cfg,
        logging=log_cfg,
        evaluation=eval_cfg,
        adaptation=adapt_cfg,
    )
    validate_config(cfg)
    return cfg


def test_trainer_image_pipeline_riac_logs_intrinsic(tmp_path: Path):
    run_dir = tmp_path / "run_riac_img"
    cfg = _make_cfg(method="riac", eta=0.1, total_steps_per_update=6, minibatches=1, epochs=1)
    out = run_train(cfg, total_steps=12, run_dir=run_dir, resume=False)

    csv_path = out / "logs" / "scalars.csv"
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    assert "r_int_mean" in set(lines[0].split(","))
