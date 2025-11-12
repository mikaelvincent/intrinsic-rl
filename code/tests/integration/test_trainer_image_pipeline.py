import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

from irl.cfg import Config, validate_config
from irl.trainer import train as run_train


# ----- Tiny uint8 RGB image env (Discrete actions) -----


class _DummyImageEnv(gym.Env):
    """Small deterministic image env to exercise the image pipeline end-to-end.

    - Observation: uint8 RGB image [H,W,C] in [0..255]
    - Action: Discrete(3)
    - Reward: +0.1 per step
    - Episode terminates at t >= 5
    """

    metadata = {"render_modes": []}

    def __init__(self, h: int = 16, w: int = 16, seed: int | None = None) -> None:
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
        info = {}
        return obs, reward, terminated, truncated, info


# Register once (idempotent across test runs)
try:
    register(id="DummyImage-v0", entry_point=_DummyImageEnv)
except Exception:
    # Already registered in this process
    pass


def _make_cfg(
    *,
    method: str,
    total_steps_per_update: int = 8,
    minibatches: int = 2,
    epochs: int = 1,
    eta: float = 0.0,
) -> Config:
    """Build a small Config that uses the dummy image env."""
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
    eval_cfg = replace(base.evaluation, interval_steps=10_000, episodes=1)  # effectively disabled
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


def _read_csv_header_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        rows = list(r)
    header = rows[0] if rows else []
    return header, rows[1:]


def test_trainer_image_pipeline_vanilla_smoke(tmp_path: Path):
    """End-to-end smoke: trainer → compute_gae → ppo_update on uint8 images (vanilla)."""
    run_dir = tmp_path / "run_vanilla_img"
    cfg = _make_cfg(method="vanilla", eta=0.0, total_steps_per_update=8, minibatches=2, epochs=1)
    out = run_train(cfg, total_steps=16, run_dir=run_dir, resume=False)

    # Scalars CSV should exist and have expected basics
    csv_path = out / "logs" / "scalars.csv"
    assert csv_path.exists(), "scalars.csv not written"
    header, rows = _read_csv_header_rows(csv_path)

    # Core metrics from the trainer loop
    assert "reward_mean" in header
    assert "reward_total_mean" in header
    assert "entropy_last" in header
    assert "entropy_update_mean" in header
    assert len(rows) >= 1  # at least one logged step


def test_trainer_image_pipeline_riac_intrinsic_smoke(tmp_path: Path):
    """End-to-end smoke with an image intrinsic (RIAC) to exercise ICM/CNN path and intrinsic logging."""
    run_dir = tmp_path / "run_riac_img"
    # Small steps; enable intrinsic via eta>0
    cfg = _make_cfg(method="riac", eta=0.1, total_steps_per_update=6, minibatches=1, epochs=1)
    out = run_train(cfg, total_steps=12, run_dir=run_dir, resume=False)

    csv_path = out / "logs" / "scalars.csv"
    assert csv_path.exists(), "scalars.csv not written"
    header, rows = _read_csv_header_rows(csv_path)

    # Intrinsic logging present
    assert "r_int_mean" in header, "expected intrinsic mean in logs"
    # RIAC-specific update metrics should be namespaced
    assert "riac_loss_total" in header
    assert "riac_loss_forward" in header
    assert "riac_loss_inverse" in header
    assert "riac_intrinsic_mean" in header
    assert len(rows) >= 1
