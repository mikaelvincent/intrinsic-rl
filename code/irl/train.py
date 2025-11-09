"""PPO training CLI — thin wrapper around irl.trainer.train().

This keeps the public CLI stable while moving the heavy training logic into the irl.trainer subpackage to meet Sprint 5
refactor goals.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import typer

from irl.cfg import Config, load_config, validate_config
from irl.trainer import train as run_train

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("train")
def cli_train(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration (see configs/*.yaml).",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    total_steps: int = typer.Option(
        10_000,
        "--total-steps",
        "-n",
        help="Total environment steps to run (across all envs).",
    ),
    run_dir: Optional[Path] = typer.Option(
        None,
        "--run-dir",
        help="Run directory for logs and checkpoints (auto if omitted).",
    ),
    method: Optional[str] = typer.Option(
        None,
        "--method",
        help="Override method in config (vanilla|icm|rnd|ride|riac|proposed). Defaults to 'vanilla' if no config.",
    ),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        "-e",
        help="Override Gymnasium environment id (e.g., Ant-v4, HalfCheetah-v4, Humanoid-v4).",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help='Override device, e.g., "cpu" or "cuda:0". (Defaults to config value or CPU.)',
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Resume from the latest checkpoint in --run-dir if available (verifies config hash).",
    ),
) -> None:
    """Launch PPO training with optional intrinsic rewards."""
    if config is not None:
        cfg = load_config(str(config))
    else:
        cfg = Config()
    # If no config and no method override, default to vanilla for smoke runs.
    if config is None and method is None:
        method = "vanilla"

    # Apply CLI overrides while preserving other config fields.
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
