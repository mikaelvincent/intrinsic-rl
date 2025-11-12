"""Allow `python -m irl` to run a tiny helper."""


def main() -> None:
    msg = (
        "intrinsic-rl utilities\n"
        "This package bundles PPO training with intrinsic modules (ICM, RND, RIDE, RIAC, proposed).\n"
        "Available CLIs: irl-train, irl-eval, irl-plot, irl-sweep. See the README for usage examples.\n"
        "Import `irl` or run `irl-train --help` to get started."
    )
    print(msg)


if __name__ == "__main__":
    main()
