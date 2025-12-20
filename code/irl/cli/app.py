from __future__ import annotations

from collections.abc import Sequence

import typer
from typer.main import get_command

from irl.benchmarks.cli import cli_bench
from irl.experiments import app as suite_app

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")
app.command("bench")(cli_bench)
app.add_typer(suite_app, name="suite")


def run(argv: Sequence[str] | None = None, *, prog_name: str | None = None) -> None:
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
