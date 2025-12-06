"""Entry point for ``python -m irl``.

Running the package as a module prints a short overview of the available
command-line tools and where to find configuration defaults.
"""


def main() -> None:
    """Print a brief help message for :mod:`irl` utilities."""
    msg = (
        "intrinsic-rl utilities\n"
        "Available entry points: irl-train, irl-eval, irl-plot, irl-sweep.\n"
        "Configuration parsing lives in irl.cfg.loader; inspect that module for defaults.\n"
        "Import `irl` in Python or run `irl-<command> --help` to explore."
    )
    print(msg)


if __name__ == "__main__":
    main()
