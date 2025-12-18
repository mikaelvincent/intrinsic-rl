from __future__ import annotations

from collections.abc import Sequence

import typer
from typer.main import get_command

from irl.benchmarks.cli import cli_bench
from irl.eval import cli_eval
from irl.train import cli_train

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")
app.command("train")(cli_train)
app.command("eval")(cli_eval)
app.command("bench")(cli_bench)

from irl.experiments import app as suite_app
from irl.multiseed.cli import app as sweep_app
from irl.visualization.cli import app as plot_app

app.add_typer(suite_app, name="suite")
app.add_typer(sweep_app, name="sweep")
app.add_typer(plot_app, name="plot")


def run(argv: Sequence[str] | None = None, *, prog_name: str | None = None) -> None:
    # Use click.Command.main so wrappers can override prog_name consistently.
    cmd = get_command(app)
    cmd.main(args=None if argv is None else list(argv), prog_name=prog_name)


def dispatch(prefix: str, argv: Sequence[str] | None = None, *, prog_name: str | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if args and str(args[0]).strip() == str(prefix):
        args = args[1:]
    run([str(prefix), *args], prog_name=prog_name)


def main(argv: list[str] | None = None) -> None:
    run(argv, prog_name="intrinsic-rl")


if __name__ == "__main__":
    main()
