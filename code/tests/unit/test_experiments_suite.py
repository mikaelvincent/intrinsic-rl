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

    captured: list[dict[str, int | bool]] = []

    import irl.experiments.training as training_module

    def fake_run_train(cfg, *args, **kwargs):
        captured.append(
            {
                "vec_envs": int(cfg.env.vec_envs),
                "async_vector": bool(getattr(cfg.env, "async_vector", False)),
                "total_steps": int(kwargs["total_steps"]),
            }
        )

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

    by_vec = {int(r["vec_envs"]): r for r in captured}
    assert by_vec[1]["total_steps"] == 24
    assert by_vec[2]["total_steps"] == 8
    assert by_vec[2]["async_vector"] is False


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
