"""Allow `python -m irl` to run a tiny helper."""


def main() -> None:
    msg = (
        "intrinsic-rl (skeleton)\n"
        "This is a Sprint 0 package scaffold. CLI commands will land in later steps.\n"
        "Modules present: cfg, envs, models, intrinsic, algo, data.\n"
        "Try importing `irl` in Python to verify installation."
    )
    print(msg)


if __name__ == "__main__":
    main()
