"""Allow `python -m irl` to run a tiny helper."""


def main() -> None:
    msg = (
        "intrinsic-rl utilities\n"
        "Available entry points: irl-train, irl-eval, irl-plot, irl-sweep.\n"
        "See the project README for configuration tips and examples.\n"
        "Import `irl` in Python or run `irl-<command> --help` to explore."
    )
    print(msg)


if __name__ == "__main__":
    main()
