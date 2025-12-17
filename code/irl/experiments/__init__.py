from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import typer

from irl.cli.common import QUICK_EPISODES
from .evaluation import run_eval_suite
from .plotting import run_plots_suite
from .training import run_training_suite
from .validation import run_validate_results
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
    episodes: int = typer.Option(20, "--episodes", "-e"),
    device: str = typer.Option("cpu", "--device", "-d"),
    policy: str = typer.Option("mode", "--policy", "-p"),
    quick: bool = typer.Option(False, "--quick/--no-quick"),
    strict_coverage: bool = typer.Option(False, "--strict-coverage/--no-strict-coverage"),
    strict_step_parity: bool = typer.Option(False, "--strict-step-parity/--no-strict-step-parity"),
) -> None:
    n_eps = int(episodes)
    if quick:
        n_eps = min(n_eps, QUICK_EPISODES)

    run_eval_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        episodes=n_eps,
        device=device,
        policy_mode=str(policy),
        strict_coverage=bool(strict_coverage),
        strict_step_parity=bool(strict_step_parity),
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
    policy: str = typer.Option("mode", "--policy", "-p"),
    seed: List[int] = typer.Option([100], "--seed", "-s"),
    max_steps: int = typer.Option(1000, "--max-steps"),
    fps: int = typer.Option(30, "--fps"),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite"),
) -> None:
    run_video_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        device=device,
        policy_mode=str(policy),
        eval_seeds=seed,
        max_steps=int(max_steps),
        fps=int(fps),
        overwrite=bool(overwrite),
    )


@app.command("validate-results")
def cli_validate_results(
    runs_root: Path = typer.Option(
        Path("runs_suite"),
        "--runs-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    results_dir: Path = typer.Option(
        Path("results_suite"),
        "--results-dir",
        "-o",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    strict: bool = typer.Option(True, "--strict/--no-strict"),
) -> None:
    ok = run_validate_results(runs_root=runs_root, results_dir=results_dir, strict=bool(strict))
    if not ok:
        raise typer.Exit(code=1)


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
    episodes: int = typer.Option(20, "--episodes", "-e"),
    results_dir: Path = typer.Option(Path("results_suite"), "--results-dir", "-o"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m"),
    smooth: int = typer.Option(5, "--smooth", "-w"),
    shade: bool = typer.Option(True, "--shade/--no-shade"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    auto_async: bool = typer.Option(False, "--auto-async/--no-auto-async"),
    policy: str = typer.Option("mode", "--policy", "-p"),
    quick: bool = typer.Option(False, "--quick/--no-quick"),
    strict_coverage: bool = typer.Option(False, "--strict-coverage/--no-strict-coverage"),
    strict_step_parity: bool = typer.Option(False, "--strict-step-parity/--no-strict-step-parity"),
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

    n_eps = int(episodes)
    if quick:
        n_eps = min(n_eps, QUICK_EPISODES)

    run_eval_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        episodes=n_eps,
        device=eval_device,
        policy_mode=str(policy),
        strict_coverage=bool(strict_coverage),
        strict_step_parity=bool(strict_step_parity),
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
        policy_mode=str(policy).strip().lower(),
        eval_seeds=[100],
        max_steps=1000,
        fps=30,
        overwrite=False,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
