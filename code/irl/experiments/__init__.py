"""Experiment suite runner for intrinsic-rl.

This package provides a Typer-based CLI and helpers to:

  * Train all eligible configuration files under a configs/ tree.
  * Evaluate the latest checkpoint for each run directory.
  * Generate simple per-environment overlay plots.
  * Generate split-screen comparison videos.

Typical usage from the repo's `code/` directory:

    # One-shot: train all configs, then eval + plots + videos
    python -m irl.experiments full

Or individual stages:

    python -m irl.experiments train
    python -m irl.experiments eval
    python -m irl.experiments plots
    python -m irl.experiments videos
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import typer  # noqa: E402
import torch  # noqa: E402

from irl.cfg import load_config
from irl.cfg.schema import Config
from irl.trainer import train as run_train  # re-exported only for tests/monkeypatching

from .training import run_training_suite
from .evaluation import run_eval_suite
from .plotting import run_plots_suite
from .videos import run_video_suite

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("train")
def cli_train(
    configs_dir: Path = typer.Option(
        Path("configs"),
        "--configs-dir",
        "-c",
        help="Root directory containing YAML configs (scanned recursively).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    include: List[str] = typer.Option(
        [],
        "--include",
        "-i",
        help="Glob(s) relative to configs_dir to select configs (e.g. 'mountaincar_*.yaml'). "
        "If omitted, all *.yaml/ *.yml files are used.",
    ),
    exclude: List[str] = typer.Option(
        [],
        "--exclude",
        "-x",
        help="Glob(s) relative to configs_dir to exclude (e.g. 'mujoco/*_debug.yaml').",
    ),
    total_steps: int = typer.Option(
        150_000,
        "--total-steps",
        "-t",
        help="Default target environment steps per run (overridden by exp.total_steps in config if present).",
    ),
    runs_root: Path = typer.Option(
        Path("runs_suite"),
        "--runs-root",
        help="Root directory where suite run subdirectories will be created.",
    ),
    seed: List[int] = typer.Option(
        [],
        "--seed",
        "-r",
        help="Override seeds to run for every config (repeatable). "
        "If omitted, each config's own seed is used.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help='Override device for training (e.g. "cpu" or "cuda:0"). "Defaults to each config\'s device field.',
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="When enabled, resume from existing checkpoints and skip runs that already reached total_steps.",
    ),
    auto_async: bool = typer.Option(
        False,
        "--auto-async/--no-auto-async",
        help="When enabled, auto-enable async vector envs for configs requesting multiple environments.",
    ),
) -> None:
    """Train all eligible configs into a deterministic runs_root."""
    run_training_suite(
        configs_dir=configs_dir,
        include=include,
        exclude=exclude,
        total_steps=total_steps,
        runs_root=runs_root,
        seeds=seed,
        device=device,
        resume=resume,
        auto_async=auto_async,
    )


@app.command("eval")
def cli_eval(
    runs_root: Path = typer.Option(
        Path("runs_suite"),
        "--runs-root",
        help="Root directory that holds suite run subdirectories.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    results_dir: Path = typer.Option(
        Path("results_suite"),
        "--results-dir",
        "-o",
        help="Directory to write summary_raw.csv and summary.csv.",
    ),
    episodes: int = typer.Option(
        5,
        "--episodes",
        "-e",
        help="Number of evaluation episodes per checkpoint.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help='Device to use for evaluation (e.g. "cpu" or "cuda:0").',
    ),
) -> None:
    """Evaluate latest checkpoints for all suite runs."""
    run_eval_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        episodes=episodes,
        device=device,
    )


@app.command("plots")
def cli_plots(
    runs_root: Path = typer.Option(
        Path("runs_suite"),
        "--runs-root",
        help="Root directory that holds suite run subdirectories.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    results_dir: Path = typer.Option(
        Path("results_suite"),
        "--results-dir",
        "-o",
        help="Directory where plots/ will be created.",
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        "-m",
        help="Scalar metric to plot. If omitted, generates standard paper plots (Extrinsic, Total, Ablation).",
    ),
    smooth: int = typer.Option(
        5,
        "--smooth",
        "-w",
        help="Moving-average window (in logged points).",
    ),
    shade: bool = typer.Option(
        True,
        "--shade/--no-shade",
        help="Shade ±1 std band on overlay plots when ≥2 runs are available.",
    ),
) -> None:
    """Generate per-environment overlay plots from suite runs."""
    run_plots_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        metric=metric,
        smooth=smooth,
        shade=shade,
    )


@app.command("videos")
def cli_videos(
    runs_root: Path = typer.Option(
        Path("runs_suite"),
        "--runs-root",
        help="Root directory that holds suite run subdirectories.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    results_dir: Path = typer.Option(
        Path("results_suite"),
        "--results-dir",
        "-o",
        help="Directory where videos/ will be created.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help='Device to use for rendering (e.g. "cpu" or "cuda:0").',
    ),
    baseline: str = typer.Option(
        "vanilla",
        "--baseline",
        "-b",
        help="Baseline method name (left side of video).",
    ),
    method: str = typer.Option(
        "proposed",
        "--method",
        "-m",
        help="Target method name (right side of video).",
    ),
) -> None:
    """Generate side-by-side comparison videos for all environments."""
    run_video_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        device=device,
        baseline=baseline,
        method=method,
    )


@app.command("full")
def cli_full(
    configs_dir: Path = typer.Option(
        Path("configs"),
        "--configs-dir",
        "-c",
        help="Root directory containing YAML configs (scanned recursively).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    include: List[str] = typer.Option(
        [],
        "--include",
        "-i",
        help="Glob(s) relative to configs_dir to select configs (e.g. 'mountaincar_*.yaml'). "
        "If omitted, all *.yaml/ *.yml files are used.",
    ),
    exclude: List[str] = typer.Option(
        [],
        "--exclude",
        "-x",
        help="Glob(s) relative to configs_dir to exclude.",
    ),
    total_steps: int = typer.Option(
        150_000,
        "--total-steps",
        "-t",
        help="Default step budget per run (overridden by exp.total_steps in config if present).",
    ),
    runs_root: Path = typer.Option(
        Path("runs_suite"),
        "--runs-root",
        help="Root directory where suite run subdirectories will be created.",
    ),
    seed: List[int] = typer.Option(
        [],
        "--seed",
        "-r",
        help="Override seeds to run for every config (repeatable). "
        "If omitted, each config's own seed is used.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help='Override device for training/evaluation (e.g. "cpu" or "cuda:0"). "Defaults to each config\'s device field for training, and "cpu" for eval if unset.',
    ),
    episodes: int = typer.Option(
        5,
        "--episodes",
        "-e",
        help="Number of evaluation episodes per checkpoint.",
    ),
    results_dir: Path = typer.Option(
        Path("results_suite"),
        "--results-dir",
        "-o",
        help="Directory to write summaries and plots.",
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        "-m",
        help="Scalar metric to plot. If omitted, generates standard paper plots.",
    ),
    smooth: int = typer.Option(
        5,
        "--smooth",
        "-w",
        help="Moving-average window (in logged points).",
    ),
    shade: bool = typer.Option(
        True,
        "--shade/--no-shade",
        help="Shade ±std band on overlay plots when ≥2 runs are available.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="When enabled, resume from existing checkpoints and skip runs that already reached total_steps.",
    ),
    auto_async: bool = typer.Option(
        False,
        "--auto-async/--no-auto-async",
        help="When enabled, auto-enable async vector envs for configs requesting multiple environments.",
    ),
) -> None:
    """Run training, evaluation, plotting, and video generation in one shot."""
    run_training_suite(
        configs_dir=configs_dir,
        include=include,
        exclude=exclude,
        total_steps=total_steps,
        runs_root=runs_root,
        seeds=seed,
        device=device,
        resume=resume,
        auto_async=auto_async,
    )

    if device is None:
        eval_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        eval_device = device

    run_eval_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        episodes=episodes,
        device=eval_device,
    )

    run_plots_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        metric=metric,
        smooth=smooth,
        shade=shade,
    )

    run_video_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        device=eval_device,
        baseline="vanilla",
        method="proposed",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
