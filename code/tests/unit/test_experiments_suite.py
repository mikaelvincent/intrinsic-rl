from __future__ import annotations

from pathlib import Path

import pytest
import torch

from irl.experiments import run_plots_suite, run_training_suite


def _write_cfg(configs_dir: Path, filename: str, text: str) -> Path:
    configs_dir.mkdir(parents=True, exist_ok=True)
    p = configs_dir / filename
    p.write_text(text.lstrip(), encoding="utf-8")
    return p


def test_training_suite_prefers_exp_total_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_training_suite_skips_when_up_to_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs_dir = tmp_path / "configs"
    cfg_path = _write_cfg(
        configs_dir,
        "mc.yaml",
        """
seed: 1
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
  total_steps: 16
""",
    )

    runs_root = tmp_path / "runs_suite"
    run_dir = runs_root / f"vanilla__MountainCar-v0__seed1__{cfg_path.stem}"
    ckpt_latest = run_dir / "checkpoints" / "ckpt_latest.pt"
    ckpt_latest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 16}, ckpt_latest)

    import irl.experiments.training as training_module

    def fail_run_train(*args, **kwargs):
        raise AssertionError("run_train should not be called when up to date")

    monkeypatch.setattr(training_module, "run_train", fail_run_train)

    run_training_suite(
        configs_dir=configs_dir,
        include=[],
        exclude=[],
        total_steps=100,
        runs_root=runs_root,
        seeds=[1],
        device="cpu",
        resume=True,
    )


@pytest.mark.parametrize("auto_async, expected", [(False, False), (True, True)])
def test_training_suite_auto_async_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auto_async: bool, expected: bool
) -> None:
    configs_dir = tmp_path / "configs"
    _write_cfg(
        configs_dir,
        "mc_vec.yaml",
        """
seed: 1
device: "cpu"
method: "vanilla"
env:
  id: "MountainCar-v0"
  vec_envs: 2
  async_vector: false
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
  total_steps: 16
""",
    )

    captured: dict[str, bool] = {}

    import irl.experiments.training as training_module

    def fake_run_train(cfg, *args, **kwargs):
        captured["async_vector"] = bool(cfg.env.async_vector)

    monkeypatch.setattr(training_module, "run_train", fake_run_train)

    run_training_suite(
        configs_dir=configs_dir,
        include=[],
        exclude=[],
        total_steps=16,
        runs_root=tmp_path / "runs_suite",
        seeds=[1],
        device="cpu",
        resume=False,
        auto_async=auto_async,
    )

    assert captured["async_vector"] is expected


def test_plots_suite_discovers_runs_and_writes_overlay(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs_suite"
    results_dir = tmp_path / "results_suite"

    for name in ("vanilla__MountainCar-v0__seed1__A", "proposed__MountainCar-v0__seed2__B"):
        logs = runs_root / name / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "scalars.csv").write_text(
            "step,reward_total_mean\n0,0.0\n10,1.0\n",
            encoding="utf-8",
        )

    run_plots_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        metric="reward_total_mean",
        smooth=1,
        shade=False,
    )

    out = results_dir / "plots" / "MountainCar-v0__overlay_reward_total_mean.png"
    assert out.exists()
