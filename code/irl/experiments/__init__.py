from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import typer

from .evaluation import run_eval_suite
from .plotting import run_plots_suite
from .training import run_training_suite
from .videos import run_video_suite

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("train")
def cli_train(
    configs_dir: Path = typer.Option(
        Path("configs"),
        "--configs-dir",
        "-c",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    include: List[str] = typer.Option([], "--include", "-i"),
    exclude: List[str] = typer.Option([], "--exclude", "-x"),
    total_steps: int = typer.Option(150_000, "--total-steps", "-t"),
    runs_root: Path = typer.Option(Path("runs_suite"), "--runs-root"),
    seed: List[int] = typer.Option([], "--seed", "-r"),
    device: Optional[str] = typer.Option(None, "--device", "-d"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    auto_async: bool = typer.Option(False, "--auto-async/--no-auto-async"),
) -> None:
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
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    results_dir: Path = typer.Option(Path("results_suite"), "--results-dir", "-o"),
    episodes: int = typer.Option(5, "--episodes", "-e"),
    device: str = typer.Option("cpu", "--device", "-d"),
) -> None:
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
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    results_dir: Path = typer.Option(Path("results_suite"), "--results-dir", "-o"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m"),
    smooth: int = typer.Option(5, "--smooth", "-w"),
    shade: bool = typer.Option(True, "--shade/--no-shade"),
) -> None:
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
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    results_dir: Path = typer.Option(Path("results_suite"), "--results-dir", "-o"),
    device: str = typer.Option("cpu", "--device", "-d"),
    baseline: str = typer.Option("vanilla", "--baseline", "-b"),
    method: str = typer.Option("proposed", "--method", "-m"),
) -> None:
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
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    include: List[str] = typer.Option([], "--include", "-i"),
    exclude: List[str] = typer.Option([], "--exclude", "-x"),
    total_steps: int = typer.Option(150_000, "--total-steps", "-t"),
    runs_root: Path = typer.Option(Path("runs_suite"), "--runs-root"),
    seed: List[int] = typer.Option([], "--seed", "-r"),
    device: Optional[str] = typer.Option(None, "--device", "-d"),
    episodes: int = typer.Option(5, "--episodes", "-e"),
    results_dir: Path = typer.Option(Path("results_suite"), "--results-dir", "-o"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m"),
    smooth: int = typer.Option(5, "--smooth", "-w"),
    shade: bool = typer.Option(True, "--shade/--no-shade"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    auto_async: bool = typer.Option(False, "--auto-async/--no-auto-async"),
) -> None:
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

    eval_device = "cuda:0" if device is None and torch.cuda.is_available() else (device or "cpu")

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
