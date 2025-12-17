from irl.multiseed.evaluation import (
    RunResult,
    _aggregate,
    _find_latest_ckpt,
    _write_raw_csv,
    _write_summary_csv,
    app,
    cli_eval_many,
    cli_stats,
)

__all__ = (
    "RunResult",
    "_aggregate",
    "_find_latest_ckpt",
    "_write_raw_csv",
    "_write_summary_csv",
    "app",
    "cli_eval_many",
    "cli_stats",
    "main",
)


def main(argv: list[str] | None = None) -> None:
    from irl.cli.app import dispatch

    dispatch("sweep", argv, prog_name="irl-sweep")
