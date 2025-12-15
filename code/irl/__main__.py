def main() -> None:
    msg = (
        "intrinsic-rl utilities\n"
        "Available entry points: irl-train, irl-eval, irl-plot, irl-sweep.\n"
        "Configuration parsing lives in irl.cfg.loader; inspect that module for defaults.\n"
        "Import `irl` in Python or run `irl-<command> --help` to explore."
    )
    print(msg)


if __name__ == "__main__":
    main()
