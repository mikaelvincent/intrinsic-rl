"""Allow `python -m irl` to run a tiny helper."""


def main() -> None:
    msg = (
        "intrinsic-rl\n"
        "Command-line entry point for the intrinsic RL toolkit.\n"
        "Available CLIs: irl-train, irl-eval, irl-plot, irl-sweep.\n"
        "See the README for quickstart and configuration guidance."
    )
    print(msg)


if __name__ == "__main__":
    main()
