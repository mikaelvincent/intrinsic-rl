"""Typer CLI for learning-curve and summary plotting.

This is the concrete implementation behind the :mod:`irl.plot` entry
points. It relies on the data helpers and figure utilities provided by
``irl.visualization.data`` and ``irl.visualization.figures``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# Use a non-interactive backend for headless environments
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from irl.utils.checkpoint import atomic_replace
from .data import (
    AggregateResult,
    _aggregate_runs,
    _expand_run_dirs,
    _parse_run_name,
    _ensure_parent,
)
from .figures import plot_normalized_summary, plot_trajectory_heatmap

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("curves")
def cli_curves(
    runs: List[str] = typer.Option(
        ...,
        "--runs",
        "-r",
        help='Glob(s) to run directories (e.g., "runs/proposed__BipedalWalker*"). '
        "You may pass this option multiple times.",
    ),
    metric: str = typer.Option(
        "reward_total_mean",
        "--metric",
        "-m",
        help="Column to plot from scalars.csv (default: reward_total_mean).",
    ),
    smooth: int = typer.Option(
        1,
        "--smooth",
        "-s",
        help="Moving-average window (in logged points). 1 disables smoothing.",
    ),
    shade: bool = typer.Option(
        False,
        "--shade/--no-shade",
        help="Fill ±std as a translucent band (enabled only if ≥2 runs).",
    ),
    label: Optional[str] = typer.Option(
        None, "--label", "-l", help="Legend label (defaults to method/env hint if available)."
    ),
    out: Path = typer.Option(
        Path("results/curve.png"),
        "--out",
        "-o",
        help="Output image path (PNG).",
        dir_okay=False,
    ),
) -> None:
    """Plot an aggregate learning curve (mean ± std) for one method/group."""
    run_dirs = _expand_run_dirs(runs)
    if not run_dirs:
        raise typer.BadParameter("No matching run directories found for --runs.")

    agg = _aggregate_runs(run_dirs, metric=metric, smooth=smooth)
    if agg.n_runs == 0:
        typer.echo("[warn] No valid data found for metric in specified runs.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    lbl = label or f"{(agg.method_hint or '').strip()} {(agg.env_hint or '').strip()}".strip()
    if not lbl:
        lbl = f"{metric} (n={agg.n_runs})"
    ax.plot(agg.steps, agg.mean, label=lbl)
    if shade and agg.n_runs >= 2 and agg.std.size > 0:
        lo = agg.mean - agg.std
        hi = agg.mean + agg.std
        ax.fill_between(agg.steps, lo, hi, alpha=0.2, linewidth=0)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(metric.replace("_", " "))
    title_bits = []
    if agg.env_hint:
        title_bits.append(agg.env_hint)
    if agg.method_hint:
        title_bits.append(agg.method_hint)
    if agg.n_runs > 1:
        title_bits.append(f"(n={agg.n_runs})")
    if title_bits:
        ax.set_title(" ".join(title_bits))
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    _ensure_parent(out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    fmt = out.suffix.lstrip(".").lower() or "png"
    fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format=fmt)
    atomic_replace(tmp, out)
    plt.close(fig)
    typer.echo(f"[green]Saved[/green] {out}")


@app.command("overlay")
def cli_overlay(
    group: List[str] = typer.Option(
        ...,
        "--group",
        "-g",
        help="One group per option; value is comma-separated glob(s) "
        '(e.g., --group "runs/proposed__BipedalWalker*" '
        '--group "runs/ride__BipedalWalker*,runs/rnd__BipedalWalker*").',
    ),
    labels: Optional[List[str]] = typer.Option(
        None,
        "--labels",
        "-l",
        help="Optional labels per group (repeat to match number of --group entries).",
    ),
    metric: str = typer.Option(
        "reward_total_mean",
        "--metric",
        "-m",
        help="Column to plot from scalars.csv (default: reward_total_mean).",
    ),
    smooth: int = typer.Option(
        1,
        "--smooth",
        "-s",
        help="Moving-average window (in logged points). 1 disables smoothing.",
    ),
    shade: bool = typer.Option(
        False,
        "--shade/--no-shade",
        help="Fill ±std for each group (may look busy; only if ≥2 runs in group).",
    ),
    out: Path = typer.Option(
        Path("results/overlay.png"),
        "--out",
        "-o",
        help="Output image path (PNG).",
        dir_okay=False,
    ),
) -> None:
    """Overlay multiple aggregated learning curves (one line per group)."""
    if not group:
        raise typer.BadParameter("Provide at least one --group.")

    if labels is not None and len(labels) != len(group):
        raise typer.BadParameter("--labels count must match the number of --group entries.")

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, spec in enumerate(group):
        patterns = [p.strip() for p in spec.split(",") if p.strip()]
        run_dirs = _expand_run_dirs(patterns)
        if not run_dirs:
            typer.echo(f"[warn] No runs found for group {i+1} spec: {spec!r}, skipping.")
            continue

        agg = _aggregate_runs(run_dirs, metric=metric, smooth=smooth)
        if agg.n_runs == 0:
            continue

        lbl = (labels[i] if labels and i < len(labels) else None) or agg.method_hint or f"group-{i+1}"
        ax.plot(agg.steps, agg.mean, label=f"{lbl} (n={agg.n_runs})")
        if shade and agg.n_runs >= 2 and agg.std.size > 0:
            lo = agg.mean - agg.std
            hi = agg.mean + agg.std
            ax.fill_between(agg.steps, lo, hi, alpha=0.15, linewidth=0)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Learning Curves (mean ± std per group)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    _ensure_parent(out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    fmt = out.suffix.lstrip(".").lower() or "png"
    fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format=fmt)
    atomic_replace(tmp, out)
    plt.close(fig)
    typer.echo(f"[green]Saved[/green] {out}")


@app.command("bars")
def cli_bars(
    summary: Path = typer.Option(
        Path("results/summary.csv"),
        "--summary",
        "-s",
        help="Path to aggregated summary CSV produced by `irl.sweep eval-many`.",
        exists=True,
        dir_okay=False,
    ),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        "-e",
        help="Filter by env_id (e.g., BipedalWalker-v3). If omitted, plots all envs (grouped by env).",
    ),
    topk: int = typer.Option(
        0,
        "--topk",
        "-k",
        help="If >0 and env is omitted, keeps only the top-K (by mean_return_mean) rows per env.",
    ),
    out: Path = typer.Option(
        Path("results/bars.png"),
        "--out",
        "-o",
        help="Output image path (PNG).",
        dir_okay=False,
    ),
) -> None:
    """Create a bar chart from the sweep's aggregated summary (summary.csv)."""
    df = pd.read_csv(summary)
    required = {
        "method",
        "env_id",
        "mean_return_mean",
        "mean_return_std",
    }
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(f"Missing columns in {summary}: {sorted(missing)}")

    if env is not None:
        d = df[df["env_id"] == env].copy()
        if d.empty:
            raise typer.BadParameter(f"No rows for env_id={env!r} in {summary}.")
        d = d.sort_values("mean_return_mean", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            d["method"],
            d["mean_return_mean"],
            yerr=d["mean_return_std"],
            capsize=3,
        )
        ax.set_ylabel("Mean return")
        ax.set_xlabel("Method")
        ax.set_title(f"{env} — Mean Return (± std across seeds)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
    else:
        envs = sorted(df["env_id"].unique())
        rows: list[pd.DataFrame] = []
        for e in envs:
            d = df[df["env_id"] == e].copy()
            d = d.sort_values("mean_return_mean", ascending=False)
            if topk and topk > 0:
                d = d.head(int(topk))
            rows.append(d)

        n_envs = len(rows)
        fig, axes = plt.subplots(n_envs, 1, figsize=(9, max(4, 3 * n_envs)), squeeze=False)
        for ax, d in zip(axes[:, 0], rows):
            ax.bar(
                d["method"],
                d["mean_return_mean"],
                yerr=d["mean_return_std"],
                capsize=3,
            )
            env_name = str(d["env_id"].iloc[0]) if not d.empty else ""
            ax.set_title(f"{env_name} — Mean Return (± std)")
            ax.set_ylabel("Mean return")
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", labelrotation=30)
        axes[-1, 0].set_xlabel("Method")
        fig.tight_layout()

    _ensure_parent(out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    fmt = out.suffix.lstrip(".").lower() or "png"
    fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format=fmt)
    atomic_replace(tmp, out)
    plt.close("all")
    typer.echo(f"[green]Saved[/green] {out}")


@app.command("bars-normalized")
def cli_bars_normalized(
    summary: Path = typer.Option(
        Path("results/summary.csv"),
        "--summary",
        "-s",
        help="Path to aggregated summary CSV produced by `irl.sweep eval-many`.",
        exists=True,
        dir_okay=False,
    ),
    out: Path = typer.Option(
        Path("results/normalized_scores.png"),
        "--out",
        "-o",
        help="Output image path (PNG).",
        dir_okay=False,
    ),
    highlight: str = typer.Option(
        "proposed",
        "--highlight",
        help="Method name to highlight with a distinct color.",
    ),
) -> None:
    """Create a grouped bar chart with Min-Max normalized scores per environment.

    Visualizes relative performance where 0.0 is the worst method and 1.0 is the best method
    for each environment.
    """
    plot_normalized_summary(summary, out, highlight_method=highlight)
    typer.echo(f"[green]Saved normalized bars[/green] to {out}")


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover - CLI entry
    """CLI entry point forwarding to Typer app."""
    app()
