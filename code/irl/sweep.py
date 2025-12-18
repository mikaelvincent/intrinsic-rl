from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    from irl.cli.app import dispatch

    dispatch("sweep", argv, prog_name="irl-sweep")
