"""Public multi-seed sweep facade for intrinsic-rl.

This module keeps the public :mod:`irl.sweep` import path and console
entry point stable while delegating the full implementation to
:mod:`irl.multiseed.evaluation`.
"""

from __future__ import annotations

from typing import List, Optional
from pathlib import Path

from irl.multiseed.evaluation import (
    RunResult,
    _aggregate,
    _find_latest_ckpt,
    _write_raw_csv,
    _write_summary_csv,
    app,
    cli_eval_many,
    cli_stats,
    main as _impl_main,
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


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point forwarding to :mod:`irl.multiseed.evaluation`."""
    # Typer ignores argv by default, but we keep the signature for
    # compatibility with the original `irl.sweep.main`.
    _impl_main()
