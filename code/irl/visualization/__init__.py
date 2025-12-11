"""Visualization subpackage for plotting utilities and CLIs.

This package currently hosts the learning-curve and summary plotting
implementation that backs the public :mod:`irl.plot` facade.
"""

from __future__ import annotations

# Re-export the main public helpers so they can be imported from here
# if desired (e.g., irl.visualization.plotting).
from .plotting import (
    AggregateResult,
    _aggregate_runs,
    _parse_run_name,
    plot_normalized_summary,
    plot_trajectory_heatmap,
    app,
    main,
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
