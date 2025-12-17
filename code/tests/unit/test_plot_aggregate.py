from pathlib import Path

import pytest

from irl.plot import _aggregate_runs


def test_aggregate_runs_dedups_and_falls_back(tmp_path: Path):
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
    assert agg_fb.n_runs == 1
    assert agg_fb.steps.tolist() == [0, 1000]

    with pytest.warns(UserWarning, match="gate_rate"):
        agg_missing = _aggregate_runs([run_dir], metric="gate_rate", smooth=1)
    assert agg_missing.n_runs == 0
    assert agg_missing.steps.size == 0
