from irl.plot import _aggregate_runs


def test_aggregate_dedups_duplicate_steps(tmp_path):
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
