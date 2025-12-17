from irl.visualization.plotting import (
    AggregateResult,
    _aggregate_runs,
    app,
    plot_normalized_summary,
    plot_trajectory_heatmap,
)

__all__ = (
    "AggregateResult",
    "_aggregate_runs",
    "plot_normalized_summary",
    "plot_trajectory_heatmap",
    "app",
    "main",
)


def main(argv: list[str] | None = None) -> None:
    from irl.cli.app import dispatch

    dispatch("plot", argv, prog_name="irl-plot")
