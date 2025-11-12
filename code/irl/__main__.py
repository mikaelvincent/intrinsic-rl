"""Allow `python -m irl` to run a tiny helper."""


def main() -> None:
    msg = (
        "intrinsic-rl toolkit\n"
        "Command-line entry points: irl-train, irl-eval, irl-plot, irl-sweep.\n"
        "See the README for quickstart instructions and configuration notes.\n"
        "Try importing `irl` in Python to explore the modules."
    )
    print(msg)


if __name__ == "__main__":
    main()
