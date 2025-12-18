from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from irl.cli.common import QUICK_EPISODES, validate_policy_mode
from irl.pipelines.eval import EvalCheckpoint, evaluate_checkpoints
from irl.results.summary import RunResult, _aggregate, _write_raw_csv, _write_summary_csv
from irl.stats_utils import bootstrap_ci, mannwhitney_u
from .results import _read_summary_raw, _values_for_method
from .run_discovery import _normalize_inputs

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("eval-many")
def cli_eval_many(
    runs: Optional[List[str]] = typer.Option(None, "--runs", "-r"),
    ckpt: Optional[List[Path]] = typer.Option(
        None,
        "--ckpt",
        "-k",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    episodes: int = typer.Option(20, "--episodes", "-n"),
    device: str = typer.Option("cpu", "--device", "-d"),
    policy: str = typer.Option("mode", "--policy", "-p"),
    quick: bool = typer.Option(False, "--quick/--no-quick"),
    out: Path = typer.Option(
        Path("results/summary.csv"),
        "--out",
        "-o",
        dir_okay=False,
    ),
    run_patterns: List[str] = typer.Argument([]),
) -> None:
    policy_mode = validate_policy_mode(policy, allowed=("mode", "sample"))

    n_eps = int(episodes)
    if quick:
        n_eps = min(n_eps, QUICK_EPISODES)

    all_run_patterns: List[str] = []
    if runs:
        all_run_patterns.extend(runs)
    if run_patterns:
        all_run_patterns.extend(run_patterns)

    ckpts = _normalize_inputs(all_run_patterns or None, ckpt)
    if not ckpts:
        raise typer.BadParameter("No checkpoints found. Provide --runs and/or --ckpt.")

    typer.echo(f"[info] Found {len(ckpts)} checkpoint(s). Starting evaluation...")

    specs = [EvalCheckpoint(ckpt=Path(c)) for c in ckpts]

    def _on_start(i: int, n: int, spec: EvalCheckpoint) -> None:
        typer.echo(f"  [{i}/{n}] {spec.ckpt}")

    def _on_error(_i: int, _n: int, spec: EvalCheckpoint, exc: Exception) -> None:
        typer.echo(f"[warning] Failed to evaluate {spec.ckpt}: {exc}")

    results = evaluate_checkpoints(
        specs,
        episodes=int(n_eps),
        device=str(device),
        policy_mode=str(policy_mode),
        on_start=_on_start,
        on_error=_on_error,
        skip_failures=True,
    )

    if not results:
        raise typer.Exit(code=1)

    raw_path = out.parent / "summary_raw.csv"
    _write_raw_csv(results, raw_path)

    agg = _aggregate(results)
    _write_summary_csv(agg, out)

    typer.echo("\n[green]Aggregation complete[/green]")
    typer.echo(f"Per-checkpoint results: {raw_path}")
    typer.echo(f"Aggregated summary    : {out}")
    typer.echo("\nTop-lines:")
    for row in agg:
        typer.echo(
            f"  {row['method']:>8} | {row['env_id']:<18} | seeds={row['seeds']:<10} "
            f"| mean={row['mean_return_mean']:.2f} ± {row['mean_return_std']:.2f} "
            f"(step≈{row['step_mean']})"
        )


@app.command("stats")
def cli_stats(
    summary_raw: Path = typer.Option(
        Path("results/summary_raw.csv"),
        "--summary-raw",
        "-s",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    env: str = typer.Option(..., "--env", "-e"),
    method_a: str = typer.Option(..., "--method-a", "-a"),
    method_b: str = typer.Option(..., "--method-b", "-b"),
    metric: str = typer.Option("mean_return", "--metric", "-m"),
    boot: int = typer.Option(2000, "--boot", "-B"),
    alternative: str = typer.Option("two-sided", "--alt"),
    latest_per_seed: bool = typer.Option(True, "--latest-per-seed/--all-checkpoints"),
) -> None:
    alt = alternative.strip().lower()
    if alt not in ("two-sided", "greater", "less"):
        raise typer.BadParameter("--alt must be one of: two-sided, greater, less")

    raw = _read_summary_raw(summary_raw)
    if not raw:
        raise typer.BadParameter(f"No rows parsed from {summary_raw}")

    x = _values_for_method(
        raw, env=env, method=method_a, metric=metric, latest_per_seed=latest_per_seed
    )
    y = _values_for_method(
        raw, env=env, method=method_b, metric=metric, latest_per_seed=latest_per_seed
    )

    if not x:
        raise typer.BadParameter(f"No rows for env={env!r}, method={method_a!r}")
    if not y:
        raise typer.BadParameter(f"No rows for env={env!r}, method={method_b!r}")

    res = mannwhitney_u(x, y, alternative=alt)

    def diff_mean(a, b):
        return float(float(sum(a)) / len(a) - float(sum(b)) / len(b))

    def diff_median(a, b):
        import numpy as _np

        return float(_np.median(a) - _np.median(b))

    mean_pt, mean_lo, mean_hi = (
        bootstrap_ci(x, y, diff_mean, n_boot=int(boot))
        if boot > 0
        else (res.mean_x - res.mean_y, float("nan"), float("nan"))
    )
    med_pt, med_lo, med_hi = (
        bootstrap_ci(x, y, diff_median, n_boot=int(boot))
        if boot > 0
        else (res.median_x - res.median_y, float("nan"), float("nan"))
    )

    typer.echo(f"\n[bold]Mann–Whitney U test[/bold] on {env} — metric: {metric}")
    typer.echo(f"Groups: {method_a} (n={res.n_x}) vs {method_b} (n={res.n_y})")
    typer.echo(
        f"U1={res.U1:.3f}, U2={res.U2:.3f}, U_used={res.U:.3f}, z={res.z:.3f}, "
        f"p={res.p_value:.6g}  (alt={alt})"
    )
    typer.echo(
        f"Means   : {method_a}={res.mean_x:.3f}, {method_b}={res.mean_y:.3f}  "
        f"(Δ={res.mean_x - res.mean_y:+.3f})"
    )
    typer.echo(
        f"Medians : {method_a}={res.median_x:.3f}, {method_b}={res.median_y:.3f}  "
        f"(Δ={res.median_x - res.median_y:+.3f})"
    )
    typer.echo(
        f"Effect sizes: CLES={res.cles:.3f}  Cliff's δ={res.cliffs_delta:+.3f}  (δ=2*cles-1)"
    )

    if boot > 0:
        typer.echo(f"\nBootstrap {boot}× percentile CIs (two-sided 95%):")
        typer.echo(f"  Δ mean   : {mean_pt:+.3f}  CI [{mean_lo:+.3f}, {mean_hi:+.3f}]")
        typer.echo(f"  Δ median : {med_pt:+.3f}  CI [{med_lo:+.3f}, {med_hi:+.3f}]")


def main() -> None:
    app()
