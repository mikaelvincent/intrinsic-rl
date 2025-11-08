"""Evaluate a saved policy deterministically (no intrinsic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from irl.evaluator import evaluate as run_evaluate

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("eval")
def cli_eval(
    env: str = typer.Option(..., "--env", "-e", help="Gymnasium env id (e.g., MountainCar-v0)."),
    ckpt: Path = typer.Option(
        ..., "--ckpt", "-k", help="Path to a training checkpoint file.", exists=True
    ),
    episodes: int = typer.Option(10, "--episodes", "-n", help="Number of episodes to evaluate."),
    device: str = typer.Option(
        "cpu", "--device", "-d", help='Torch device, e.g., "cpu" or "cuda:0".'
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Optional path to write aggregated results as JSON.",
        dir_okay=False,
    ),
) -> None:
    """Run evaluation episodes using mode actions and report aggregate stats."""
    summary = run_evaluate(env=env, ckpt=ckpt, episodes=episodes, device=device)

    # Per-episode lines
    for i, (ret, length) in enumerate(zip(summary["returns"], summary["lengths"]), start=1):
        typer.echo(f"Episode {i}/{summary['episodes']}: return = {ret:.2f}, length = {length}")

    # Aggregate line
    typer.echo(
        f"\n[green]Eval complete[/green] — mean return {summary['mean_return']:.2f} "
        f"± {summary['std_return']:.2f} over {summary['episodes']} episodes"
    )

    # Optional JSON dump
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        typer.echo(f"Saved summary to {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
