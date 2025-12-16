from __future__ import annotations

from .cli import app, main
from .data import (
    AggregateResult,
    _aggregate_runs,
    _dedup_paths,
    _ensure_parent,
    _expand_run_dirs,
    _parse_run_name,
    _read_scalars,
    _smooth_series,
)
from .figures import plot_normalized_summary, plot_trajectory_heatmap

__all__ = [
    "AggregateResult",
    "_aggregate_runs",
    "_dedup_paths",
    "_expand_run_dirs",
    "_parse_run_name",
    "_read_scalars",
    "_smooth_series",
    "_ensure_parent",
    "plot_normalized_summary",
    "plot_trajectory_heatmap",
    "app",
    "main",
]
