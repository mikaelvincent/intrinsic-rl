from __future__ import annotations

from pathlib import Path

import pytest

from irl.experiments import run_training_suite


def _write_cfg(configs_dir: Path, filename: str, text: str) -> Path:
    configs_dir.mkdir(parents=True, exist_ok=True)
    p = configs_dir / filename
    p.write_text(text.lstrip(), encoding="utf-8")
    return p


def test_training_suite_prefers_exp_total_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs_dir = tmp_path / "configs"
    _write_cfg(
        configs_dir,
        "mc_steps.yaml",
        """
seed: 7
device: "cpu"
method: "vanilla"
env:
  id: "MountainCar-v0"
  vec_envs: 1
ppo:
  steps_per_update: 8
  minibatches: 2
  epochs: 1
logging:
  tb: false
  csv_interval: 1
  checkpoint_interval: 100000
exp:
  deterministic: true
  total_steps: 24
""",
    )

    captured: dict[str, int] = {}

    import irl.experiments.training as training_module

    def fake_run_train(cfg, *args, **kwargs):
        captured["total_steps"] = int(kwargs["total_steps"])

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
    )

    assert captured["total_steps"] == 24
