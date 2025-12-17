from __future__ import annotations

from pathlib import Path

import pytest

from irl.experiments import run_training_suite


def _write_cfg(configs_dir: Path, filename: str, text: str) -> Path:
    configs_dir.mkdir(parents=True, exist_ok=True)
    p = configs_dir / filename
    p.write_text(text.lstrip(), encoding="utf-8")
    return p


def test_auto_async_skips_when_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs_dir = tmp_path / "configs"
    _write_cfg(
        configs_dir,
        "det_sync.yaml",
        """
seed: 7
device: "cpu"
method: "vanilla"
env:
  id: "MountainCar-v0"
  vec_envs: 2
ppo:
  steps_per_update: 8
  minibatches: 2
  epochs: 1
logging:
  csv_interval: 1
  checkpoint_interval: 100000
exp:
  deterministic: true
  total_steps: 8
""",
    )

    captured: list[bool] = []

    import irl.experiments.training as training_module

    def fake_run_train(cfg, *args, **kwargs):
        captured.append(bool(getattr(cfg.env, "async_vector", False)))

    monkeypatch.setattr(training_module, "run_train", fake_run_train)

    run_training_suite(
        configs_dir=configs_dir,
        include=[],
        exclude=[],
        total_steps=100,
        runs_root=tmp_path / "runs_suite",
        seeds=[1],
        device="cpu",
        resume=False,
        auto_async=True,
    )

    assert captured == [False]
