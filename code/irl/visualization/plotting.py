from __future__ import annotations

from .cli import app, main
from .data import (
    AggregateResult,
    aggregate_runs,
    dedup_paths,
    ensure_parent,
    expand_run_dirs,
    read_scalars,
    smooth_series,
)
from .figures import plot_normalized_summary, plot_trajectory_heatmap

__all__ = [
    "AggregateResult",
    "aggregate_runs",
    "dedup_paths",
    "expand_run_dirs",
    "read_scalars",
    "smooth_series",
    "ensure_parent",
    "plot_normalized_summary",
    "plot_trajectory_heatmap",
    "app",
    "main",
]
