from dataclasses import replace
from pathlib import Path

import typer

from irl.cli.common import resolve_default_method_for_entrypoint
from irl.cfg import Config, load_config, validate_config
from irl.trainer import train as run_train


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

    method_eff = resolve_default_method_for_entrypoint(
        config_provided=(config is not None),
        method=method,
        default_no_config="vanilla",
    )
    if method_eff is not None:
        cfg = replace(cfg, method=str(method_eff))

    if env is not None:
        cfg = replace(cfg, env=replace(cfg.env, id=str(env)))
    if device is not None:
        cfg = replace(cfg, device=str(device))

    validate_config(cfg)
    out_dir = run_train(cfg, total_steps=total_steps, run_dir=run_dir, resume=resume)
    typer.echo(f"[green]Training finished[/green]\nRun dir: {out_dir}")


def main(argv: list[str] | None = None) -> None:
    from irl.cli.app import dispatch

    dispatch("train", argv, prog_name="irl-train")


if __name__ == "__main__":
    main()
