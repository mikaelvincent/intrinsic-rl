from __future__ import annotations

from pathlib import Path

import pytest

from irl.experiments import run_training_suite
from irl.plot import _aggregate_runs


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


def test_aggregate_runs_dedups_steps_and_falls_back(tmp_path: Path):
    run_dir = tmp_path / "runs" / "vanilla__MountainCar-v0__seed1__20250101-000000"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    (logs / "scalars.csv").write_text(
        "step,reward_total_mean\n0,0.0\n0,0.5\n1000,1.5\n",
        encoding="utf-8",
    )

    agg = _aggregate_runs([run_dir], metric="reward_total_mean", smooth=1)
    assert agg.steps.tolist() == [0, 1000]
    assert agg.mean.tolist() == [0.5, 1.5]

    with pytest.warns(UserWarning, match="reward_mean"):
        agg_fb = _aggregate_runs([run_dir], metric="reward_mean", smooth=1)
    assert agg_fb.steps.tolist() == [0, 1000]
    assert agg_fb.mean.tolist() == [0.5, 1.5]
