from __future__ import annotations

from typing import List, Optional

from irl.visualization.plotting import (
    AggregateResult,
    _aggregate_runs,
    _parse_run_name,
    app,
    main as _impl_main,
    plot_normalized_summary,
    plot_trajectory_heatmap,
)

__all__ = [
    "AggregateResult",
    "_aggregate_runs",
    "_parse_run_name",
    "plot_normalized_summary",
    "plot_trajectory_heatmap",
    "app",
    "main",
]


def main(argv: Optional[List[str]] = None) -> None:
    _impl_main(argv)
