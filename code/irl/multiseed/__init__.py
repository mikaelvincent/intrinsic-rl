"""Multi-seed evaluation and statistics utilities.

This subpackage currently hosts the implementation that backs the
public :mod:`irl.sweep` facade.
"""

from __future__ import annotations

from .evaluation import (
    RunResult,
    _aggregate,
    _find_latest_ckpt,
    _write_raw_csv,
    _write_summary_csv,
    app,
    cli_eval_many,
    cli_stats,
    main,
)

__all__ = [
    "RunResult",
    "_aggregate",
    "_find_latest_ckpt",
    "_write_raw_csv",
    "_write_summary_csv",
    "app",
    "cli_eval_many",
    "cli_stats",
    "main",
]
