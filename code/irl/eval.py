import json
from pathlib import Path

import typer

from irl.cli.common import QUICK_EPISODES, validate_policy_mode
from irl.evaluator import evaluate as run_evaluate
from irl.utils.checkpoint import atomic_write_text


def _print_summary(label: str, summary: dict) -> None:
    prefix = f"{label} " if label else ""
    returns = summary.get("returns", [])
    lengths = summary.get("lengths", [])
    episodes = int(summary.get("episodes", len(returns)))
    for i, (ret, length) in enumerate(zip(returns, lengths), start=1):
        typer.echo(
            f"{prefix}Episode {i}/{episodes}: return = {float(ret):.2f}, length = {int(length)}"
        )
    typer.echo(
        f"\n{prefix}Eval complete — mean return {float(summary.get('mean_return', 0.0)):.2f} "
        f"± {float(summary.get('std_return', 0.0)):.2f} over {episodes} episodes"
    )


def cli_eval(
    env: str = typer.Option(..., "--env", "-e"),
    ckpt: Path = typer.Option(..., "--ckpt", "-k", exists=True),
    episodes: int = typer.Option(20, "--episodes", "-n"),
    device: str = typer.Option("cpu", "--device", "-d"),
    policy: str = typer.Option("mode", "--policy", "-p"),
    quick: bool = typer.Option(False, "--quick/--no-quick"),
    out: Path | None = typer.Option(None, "--out", "-o", dir_okay=False),
) -> None:
    policy_mode = validate_policy_mode(policy, allowed=("mode", "sample", "both"))

    n_eps = int(episodes)
    if quick:
        n_eps = min(n_eps, QUICK_EPISODES)

    summary = run_evaluate(env=env, ckpt=ckpt, episodes=n_eps, device=device, policy_mode=policy_mode)

    if policy_mode == "both":
        _print_summary("[Deterministic]", summary["deterministic"])
        typer.echo("")
        _print_summary("[Stochastic]", summary["stochastic"])
    else:
        _print_summary("", summary)

    if out is not None:
        atomic_write_text(out, json.dumps(summary, indent=2))
        typer.echo(f"Saved summary to {out}")


def main(argv: list[str] | None = None) -> None:
    from irl.cli.app import dispatch

    dispatch("eval", argv, prog_name="irl-eval")


if __name__ == "__main__":
    main()
