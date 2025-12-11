"""Public plotting facade for intrinsic-rl.

This module keeps the public import path :mod:`irl.plot` stable while the
implementation lives in :mod:`irl.visualization.plotting`.

Downstream code and console entry points continue to use::

    from irl import plot
    from irl.plot import _aggregate_runs, plot_normalized_summary, app, main

without needing to know about the new internal layout.
"""

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
    """CLI entry point forwarding to :mod:`irl.visualization.plotting`."""
    _impl_main(argv)
