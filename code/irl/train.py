from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer

from irl.cfg import Config, load_config, validate_config
from irl.trainer import train as run_train

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("train")
def cli_train(
    config: Path | None = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False, readable=True
    ),
    total_steps: int | None = typer.Option(None, "--total-steps", "-n"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    method: str | None = typer.Option(None, "--method"),
    env: str | None = typer.Option(None, "--env", "-e"),
    device: str | None = typer.Option(None, "--device", "-d"),
    resume: bool = typer.Option(False, "--resume/--no-resume"),
) -> None:
    cfg = load_config(str(config)) if config is not None else Config()
    if config is None and method is None:
        method = "vanilla"
    if method is not None:
        cfg = replace(cfg, method=str(method))
    if env is not None:
        cfg = replace(cfg, env=replace(cfg.env, id=str(env)))
    if device is not None:
        cfg = replace(cfg, device=str(device))
    validate_config(cfg)
    out_dir = run_train(cfg, total_steps=total_steps, run_dir=run_dir, resume=resume)
    typer.echo(f"[green]Training finished[/green]\nRun dir: {out_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
