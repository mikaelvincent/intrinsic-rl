"""Plotting utilities and CLI facade for learning curves and summaries.

This module now acts as a thin public facade that re-exports the concrete
implementations from the internal ``data``, ``figures``, and ``cli`` modules.

Existing imports such as::

    from irl.visualization.plotting import (
        AggregateResult,
        _aggregate_runs,
        _parse_run_name,
        plot_normalized_summary,
        plot_trajectory_heatmap,
        app,
        main,
    )

continue to work without modification.
"""

from __future__ import annotations

from typing import List, Optional

from .data import (
    AggregateResult,
    _aggregate_runs,
    _dedup_paths,
    _expand_run_dirs,
    _parse_run_name,
    _read_scalars,
    _smooth_series,
    _ensure_parent,
)
from .figures import plot_normalized_summary, plot_trajectory_heatmap
from .cli import app, main

__all__ = [
    # Data helpers / types
    "AggregateResult",
    "_aggregate_runs",
    "_dedup_paths",
    "_expand_run_dirs",
    "_parse_run_name",
    "_read_scalars",
    "_smooth_series",
    "_ensure_parent",
    # Figure helpers
    "plot_normalized_summary",
    "plot_trajectory_heatmap",
    # CLI surface
    "app",
    "main",
]
