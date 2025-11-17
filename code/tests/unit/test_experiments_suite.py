from __future__ import annotations

from pathlib import Path

import pytest

from irl.experiments import (
    run_training_suite,
    run_eval_suite,
    run_plots_suite,
)
from irl.utils.checkpoint import load_checkpoint


def _write_mountaincar_config(configs_dir: Path, filename: str = "mc_basic.yaml") -> Path:
    """Create a tiny MountainCar config suitable for fast suite tests."""
    configs_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = """
seed: 1
device: "cpu"
method: "vanilla"
env:
  id: "MountainCar-v0"
  vec_envs: 1
  frame_skip: 1
  domain_randomization: false
  discrete_actions: true
ppo:
  steps_per_update: 16
  minibatches: 4
  epochs: 1
  learning_rate: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.01
intrinsic:
  eta: 0.0
adaptation:
  enabled: false
evaluation:
  interval_steps: 100000
  episodes: 1
logging:
  tb: false
  csv_interval: 1
  checkpoint_interval: 16
"""
    path = configs_dir / filename
    path.write_text(cfg_text.lstrip(), encoding="utf-8")
    return path


def _single_run_dir(runs_root: Path) -> Path:
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"Expected a single run dir, found {run_dirs}"
    return run_dirs[0]


def test_suite_train_creates_run_and_checkpoint(tmp_path: Path) -> None:
    """Training suite should create a run dir, checkpoint, and scalars CSV."""
    configs_dir = tmp_path / "configs"
    _write_mountaincar_config(configs_dir)

    runs_root = tmp_path / "runs_suite"

    run_training_suite(
        configs_dir=configs_dir,
        include=[],
        exclude=[],
        total_steps=32,
        runs_root=runs_root,
        seeds=[1],
        device="cpu",
        resume=True,
    )

    run_dir = _single_run_dir(runs_root)

    ckpt_latest = run_dir / "checkpoints" / "ckpt_latest.pt"
    assert ckpt_latest.exists()

    payload = load_checkpoint(ckpt_latest, map_location="cpu")
    step = int(payload.get("step", 0))
    assert step >= 32

    csv_path = run_dir / "logs" / "scalars.csv"
    assert csv_path.exists()
    contents = csv_path.read_text(encoding="utf-8").strip().splitlines()
    # At least header + one row
    assert len(contents) >= 2
    assert contents[0].startswith("step,")


def test_suite_train_skips_when_up_to_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When resume=True and steps >= target, train suite should not call run_train again."""
    configs_dir = tmp_path / "configs"
    _write_mountaincar_config(configs_dir)
    runs_root = tmp_path / "runs_suite"
    total_steps = 16

    # First run: use real trainer to create a checkpoint at total_steps.
    run_training_suite(
        configs_dir=configs_dir,
        include=[],
        exclude=[],
        total_steps=total_steps,
        runs_root=runs_root,
        seeds=[1],
        device="cpu",
        resume=True,
    )

    run_dir = _single_run_dir(runs_root)
    ckpt_latest = run_dir / "checkpoints" / "ckpt_latest.pt"
    assert ckpt_latest.exists()
    step_before = int(load_checkpoint(ckpt_latest, map_location="cpu").get("step", 0))

    # Second run: patch run_train so the test fails if it is called.
    import irl.experiments as exp_module

    called = {"count": 0}

    def fake_run_train(*args, **kwargs):  # pragma: no cover - should not be called
        called["count"] += 1
        raise AssertionError("run_train should not be invoked when run is already up to date")

    monkeypatch.setattr(exp_module, "run_train", fake_run_train)

    run_training_suite(
        configs_dir=configs_dir,
        include=[],
        exclude=[],
        total_steps=total_steps,
        runs_root=runs_root,
        seeds=[1],
        device="cpu",
        resume=True,
    )

    assert called["count"] == 0
    step_after = int(load_checkpoint(ckpt_latest, map_location="cpu").get("step", 0))
    # Checkpoint should be unchanged by the skipped run.
    assert step_after == step_before


def test_suite_eval_and_plots_smoke(tmp_path: Path) -> None:
    """End-to-end smoke: train → eval → plots for a single config."""
    configs_dir = tmp_path / "configs"
    _write_mountaincar_config(configs_dir)

    runs_root = tmp_path / "runs_suite"
    results_dir = tmp_path / "results_suite"

    # Train a tiny run on CPU for speed.
    run_training_suite(
        configs_dir=configs_dir,
        include=[],
        exclude=[],
        total_steps=32,
        runs_root=runs_root,
        seeds=[1],
        device="cpu",
        resume=False,
    )

    # Evaluate and produce summary CSVs.
    run_eval_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        episodes=2,
        device="cpu",
    )

    raw_path = results_dir / "summary_raw.csv"
    summary_path = results_dir / "summary.csv"
    assert raw_path.exists()
    assert summary_path.exists()
    raw_text = raw_path.read_text(encoding="utf-8")
    assert "mean_return" in raw_text
    assert "env_id" in raw_text

    # Generate overlay plots from logged scalars.
    run_plots_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        metric="reward_total_mean",
        smooth=1,
        shade=False,
    )

    plots_dir = results_dir / "plots"
    assert plots_dir.exists()

    # MountainCar-v0 env tag should appear in the overlay file name.
    overlay_files = list(plots_dir.glob("MountainCar-v0__overlay_reward_total_mean.png"))
    assert overlay_files, "Expected an overlay plot for MountainCar-v0"
