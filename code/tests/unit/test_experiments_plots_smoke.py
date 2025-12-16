from pathlib import Path

import pandas as pd

from irl.plot import plot_normalized_summary


def test_plot_normalized_summary_handles_equal_ranges(tmp_path: Path) -> None:
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
