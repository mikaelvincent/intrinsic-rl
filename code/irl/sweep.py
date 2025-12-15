from __future__ import annotations

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


def main(argv: list[str] | None = None) -> None:
    _impl_main()
