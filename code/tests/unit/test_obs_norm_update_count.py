from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from irl.cfg import Config, validate_config
from irl.trainer import train as run_train
from irl.utils.checkpoint import load_checkpoint


class _ObsNormCountEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}


try:
    register(id="ObsNormCount-v0", entry_point=_ObsNormCountEnv)
except Exception:
    pass


def _make_cfg() -> Config:
    base = Config()

    env_cfg = replace(
        base.env,
        id="ObsNormCount-v0",
        vec_envs=2,
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


def test_obs_norm_counts_once_per_transition(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_obs_norm_count"
    cfg = _make_cfg()
    out_dir = run_train(cfg, total_steps=4, run_dir=run_dir, resume=False)

    ckpt_path = out_dir / "checkpoints" / "ckpt_latest.pt"
    payload = load_checkpoint(ckpt_path, map_location="cpu")

    step = int(payload.get("step", -1))
    assert step == 4

    obs_norm = payload.get("obs_norm")
    assert isinstance(obs_norm, dict)

    count = float(obs_norm.get("count", float("nan")))
    assert np.isfinite(count)
    assert abs(count - float(step)) < 1e-6
