from __future__ import annotations

from pathlib import Path

import typer


def cli_bench(
    device: str = typer.Option("cpu", "--device", "-d"),
    threads: int = typer.Option(1, "--threads", "-t"),
    seed: int = typer.Option(0, "--seed", "-s"),
    out_dir: Path = typer.Option(Path("results/benchmarks"), "--out-dir", "-o"),
    quick: bool = typer.Option(True, "--quick/--full"),
) -> None:
    from irl.benchmarks.suite import run_all_benchmarks

    payload = run_all_benchmarks(
        device=str(device),
        threads=int(threads),
        seed=int(seed),
        out_dir=Path(out_dir),
        quick=bool(quick),
    )

    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    if isinstance(outputs, dict) and outputs.get("latest_json"):
        typer.echo(f"[bench] Saved results: {outputs['latest_json']}")
