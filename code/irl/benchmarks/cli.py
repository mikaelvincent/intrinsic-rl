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

    plot_outputs: dict[str, str] = {}
    try:
        from irl.benchmarks.plots import write_benchmark_plots

        plot_outputs = write_benchmark_plots(payload, out_dir=Path(out_dir))
    except Exception as exc:
        typer.echo(f"[bench] Plot generation failed: {type(exc).__name__}: {exc}")

    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    if isinstance(outputs, dict) and outputs.get("latest_json"):
        typer.echo(f"[bench] Saved results: {outputs['latest_json']}")

    if plot_outputs.get("throughput"):
        typer.echo(f"[bench] Saved throughput plot: {plot_outputs['throughput']}")
    if plot_outputs.get("speedup"):
        typer.echo(f"[bench] Saved speedup plot: {plot_outputs['speedup']}")
