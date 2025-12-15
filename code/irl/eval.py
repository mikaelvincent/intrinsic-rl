import json
from pathlib import Path

import typer

from irl.evaluator import evaluate as run_evaluate
from irl.utils.checkpoint import atomic_write_text

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("eval")
def cli_eval(
    env: str = typer.Option(..., "--env", "-e"),
    ckpt: Path = typer.Option(..., "--ckpt", "-k", exists=True),
    episodes: int = typer.Option(10, "--episodes", "-n"),
    device: str = typer.Option("cpu", "--device", "-d"),
    out: Path | None = typer.Option(None, "--out", "-o", dir_okay=False),
) -> None:
    summary = run_evaluate(env=env, ckpt=ckpt, episodes=episodes, device=device)
    for i, (ret, length) in enumerate(zip(summary["returns"], summary["lengths"]), start=1):
        typer.echo(f"Episode {i}/{summary['episodes']}: return = {ret:.2f}, length = {length}")
    typer.echo(
        f"\n[green]Eval complete[/green] — mean return {summary['mean_return']:.2f} "
        f"± {summary['std_return']:.2f} over {summary['episodes']} episodes"
    )
    if out is not None:
        atomic_write_text(out, json.dumps(summary, indent=2))
        typer.echo(f"Saved summary to {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
