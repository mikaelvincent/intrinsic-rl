import pandas as pd

from irl.plot import _aggregate_runs, plot_normalized_summary


def test_plot_aggregate_dedups_duplicate_steps(tmp_path):
    run_dir = tmp_path / "runs" / "vanilla__MountainCar-v0__seed1__20250101-000000"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    csv_path = logs / "scalars.csv"
    csv_path.write_text("step,reward_total_mean\n0,0.0\n0,0.5\n1000,1.5\n", encoding="utf-8")

    agg = _aggregate_runs([run_dir], metric="reward_total_mean", smooth=1)

    assert agg.n_runs == 1
    assert agg.steps.tolist() == [0, 1000]
    assert agg.mean[0] == 0.5
    assert agg.mean[-1] == 1.5


def test_plot_normalized_summary_handles_equal_ranges(tmp_path):
    csv_path = tmp_path / "summary.csv"
    out_path = tmp_path / "normalized.png"

    df = pd.DataFrame(
        [
            {"env_id": "E1", "method": "A", "mean_return_mean": 10.0, "mean_return_std": 1.0},
            {"env_id": "E1", "method": "B", "mean_return_mean": 20.0, "mean_return_std": 1.0},
            {"env_id": "E2", "method": "A", "mean_return_mean": 50.0, "mean_return_std": 1.0},
            {"env_id": "E2", "method": "B", "mean_return_mean": 50.0, "mean_return_std": 1.0},
        ]
    )
    df.to_csv(csv_path, index=False)

    plot_normalized_summary(csv_path, out_path)
    assert out_path.exists()
