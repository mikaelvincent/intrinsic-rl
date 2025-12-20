from __future__ import annotations

import typer

from irl.paper_defaults import (
    BENCH_DIR,
    DEFAULT_BENCH_DEVICE,
    DEFAULT_BENCH_QUICK,
    DEFAULT_BENCH_SEED,
    DEFAULT_BENCH_THREADS,
)


def cli_bench() -> None:
    from irl.benchmarks.suite import run_all_benchmarks

    payload = run_all_benchmarks(
        device=str(DEFAULT_BENCH_DEVICE),
        threads=int(DEFAULT_BENCH_THREADS),
        seed=int(DEFAULT_BENCH_SEED),
        out_dir=BENCH_DIR,
        quick=bool(DEFAULT_BENCH_QUICK),
    )

    plot_outputs: dict[str, str] = {}
    try:
        from irl.benchmarks.plots import write_benchmark_plots

        plot_outputs = write_benchmark_plots(payload, out_dir=BENCH_DIR)
    except Exception as exc:
        typer.echo(f"[bench] Plot generation failed: {type(exc).__name__}: {exc}")

    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    if isinstance(outputs, dict) and outputs.get("latest_json"):
        typer.echo(f"[bench] Saved results: {outputs['latest_json']}")

    if plot_outputs.get("throughput"):
        typer.echo(f"[bench] Saved throughput plot: {plot_outputs['throughput']}")
    if plot_outputs.get("speedup"):
        typer.echo(f"[bench] Saved speedup plot: {plot_outputs['speedup']}")
