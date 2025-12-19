from __future__ import annotations

from pathlib import Path

from irl.visualization.figures import plot_normalized_summary


def test_plot_normalized_summary_requires_baseline(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "\n".join(
            [
                "method,env_id,mean_return_mean,n_seeds",
                "glpe,EnvA,10.0,3",
                "rnd,EnvA,8.0,3",
                "glpe,EnvB,5.0,3",
                "rnd,EnvB,6.0,3",
                "",
            ]
        ),
        encoding="utf-8",
    )

    out = tmp_path / "bars.png"
    plot_normalized_summary(
        summary,
        out,
        highlight_method=None,
        baseline_method="vanilla",
        baseline_required=True,
    )
    assert not out.exists()
