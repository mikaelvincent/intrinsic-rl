from __future__ import annotations

import typer

from irl.paper_defaults import (
    CONFIGS_DIR,
    RESULTS_DIR,
    RUNS_ROOT,
)

from .evaluation import run_eval_suite
from .plotting import run_plots_suite
from .training import run_training_suite
from .validation import run_validate_results
from .videos import run_video_suite

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("train")
def cli_train() -> None:
    run_training_suite(
        configs_dir=CONFIGS_DIR,
        runs_root=RUNS_ROOT,
    )


@app.command("eval")
def cli_eval() -> None:
    run_eval_suite(
        runs_root=RUNS_ROOT,
        results_dir=RESULTS_DIR,
    )


@app.command("plots")
def cli_plots() -> None:
    run_plots_suite(
        runs_root=RUNS_ROOT,
        results_dir=RESULTS_DIR,
    )


@app.command("videos")
def cli_videos() -> None:
    run_video_suite()


@app.command("validate")
def cli_validate(
    strict: bool = typer.Option(True, "--strict/--no-strict"),
) -> None:
    ok = run_validate_results(
        runs_root=RUNS_ROOT,
        results_dir=RESULTS_DIR,
        strict=bool(strict),
    )
    if not ok:
        raise typer.Exit(code=1)


@app.command("full")
def cli_full() -> None:
    cli_train()
    cli_eval()
    cli_plots()
    cli_videos()


def main(argv: list[str] | None = None) -> None:
    from irl.cli.app import dispatch

    dispatch("suite", argv, prog_name="irl-suite")


if __name__ == "__main__":
    main()
