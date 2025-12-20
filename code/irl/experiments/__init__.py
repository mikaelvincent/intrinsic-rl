from __future__ import annotations

import typer

from irl.paper_defaults import (
    CONFIGS_DIR,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_POLICY_MODE,
    DEFAULT_TRAIN_TOTAL_STEPS,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_MAX_STEPS,
    DEFAULT_VIDEO_POLICY_MODE,
    DEFAULT_VIDEO_SEEDS,
    RESULTS_DIR,
    RUNS_ROOT,
)

from .evaluation import run_eval_suite
from .plotting import run_plots_suite
from .training import run_training_suite
from .videos import run_video_suite

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("train")
def cli_train() -> None:
    run_training_suite(
        configs_dir=CONFIGS_DIR,
        include=[],
        exclude=[],
        total_steps=int(DEFAULT_TRAIN_TOTAL_STEPS),
        runs_root=RUNS_ROOT,
        seeds=[],
        device=None,
        resume=True,
        auto_async=False,
    )


@app.command("eval")
def cli_eval() -> None:
    run_eval_suite(
        runs_root=RUNS_ROOT,
        results_dir=RESULTS_DIR,
        episodes=int(DEFAULT_EVAL_EPISODES),
        device="cpu",
        policy_mode=str(DEFAULT_EVAL_POLICY_MODE),
    )


@app.command("plots")
def cli_plots() -> None:
    run_plots_suite(
        runs_root=RUNS_ROOT,
        results_dir=RESULTS_DIR,
        metric=None,
        smooth=5,
        shade=True,
        paper_mode=True,
    )


@app.command("videos")
def cli_videos() -> None:
    run_video_suite(
        runs_root=RUNS_ROOT,
        results_dir=RESULTS_DIR,
        device="cpu",
        policy_mode=str(DEFAULT_VIDEO_POLICY_MODE),
        eval_seeds=list(DEFAULT_VIDEO_SEEDS),
        max_steps=int(DEFAULT_VIDEO_MAX_STEPS),
        fps=int(DEFAULT_VIDEO_FPS),
        overwrite=False,
    )


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
