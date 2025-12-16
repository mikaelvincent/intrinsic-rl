from pathlib import Path

import pytest

from irl.plot import _aggregate_runs


def test_aggregate_dedups_duplicate_steps(tmp_path: Path):
    run_dir = tmp_path / "runs" / "vanilla__MountainCar-v0__seed1__20250101-000000"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    (logs / "scalars.csv").write_text(
        "step,reward_total_mean\n0,0.0\n0,0.5\n1000,1.5\n",
        encoding="utf-8",
    )

    agg = _aggregate_runs([run_dir], metric="reward_total_mean", smooth=1)

    assert agg.steps.tolist() == [0, 1000]
    assert float(agg.mean[0]) == 0.5


def test_aggregate_reward_fallback_warns_and_works(tmp_path: Path):
    run_dir = tmp_path / "runs" / "vanilla__MountainCar-v0__seed1__20250101-000000"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    (logs / "scalars.csv").write_text(
        "step,reward_total_mean\n0,1.0\n10,2.0\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="reward_mean"):
        agg = _aggregate_runs([run_dir], metric="reward_mean", smooth=1)

    assert agg.n_runs == 1
    assert agg.steps.tolist() == [0, 10]
    assert agg.mean.tolist() == [1.0, 2.0]


def test_aggregate_nonreward_missing_metric_warns_and_skips(tmp_path: Path):
    run_dir = tmp_path / "runs" / "glpe__MountainCar-v0__seed1__20250101-000000"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    (logs / "scalars.csv").write_text(
        "step,reward_total_mean,reward_mean\n0,0.0,0.0\n10,1.0,1.0\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="gate_rate"):
        agg = _aggregate_runs([run_dir], metric="gate_rate", smooth=1)

    assert agg.n_runs == 0
    assert agg.steps.size == 0
    assert agg.mean.size == 0
    assert agg.std.size == 0
