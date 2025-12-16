from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from irl.utils.checkpoint import atomic_replace
from .data import _aggregate_runs, _ensure_parent, _expand_run_dirs
from .figures import plot_normalized_summary

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


@app.command("curves")
def cli_curves(
    runs: List[str] = typer.Option(..., "--runs", "-r"),
    metric: str = typer.Option("reward_total_mean", "--metric", "-m"),
    smooth: int = typer.Option(1, "--smooth", "-s"),
    shade: bool = typer.Option(False, "--shade/--no-shade"),
    label: Optional[str] = typer.Option(None, "--label", "-l"),
    out: Path = typer.Option(Path("results/curve.png"), "--out", "-o", dir_okay=False),
) -> None:
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
    group: List[str] = typer.Option(..., "--group", "-g"),
    labels: Optional[List[str]] = typer.Option(None, "--labels", "-l"),
    metric: str = typer.Option("reward_total_mean", "--metric", "-m"),
    smooth: int = typer.Option(1, "--smooth", "-s"),
    shade: bool = typer.Option(False, "--shade/--no-shade"),
    out: Path = typer.Option(Path("results/overlay.png"), "--out", "-o", dir_okay=False),
) -> None:
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

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Learning Curves (mean per group)")
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
    summary: Path = typer.Option(Path("results/summary.csv"), "--summary", "-s", exists=True, dir_okay=False),
    env: Optional[str] = typer.Option(None, "--env", "-e"),
    topk: int = typer.Option(0, "--topk", "-k"),
    out: Path = typer.Option(Path("results/bars.png"), "--out", "-o", dir_okay=False),
) -> None:
    df = pd.read_csv(summary)
    required = {"method", "env_id", "mean_return_mean"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(f"Missing columns in {summary}: {sorted(missing)}")

    if env is not None:
        d = df[df["env_id"] == env].copy()
        if d.empty:
            raise typer.BadParameter(f"No rows for env_id={env!r} in {summary}.")
        d = d.sort_values("mean_return_mean", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(d["method"], d["mean_return_mean"])
        ax.set_ylabel("Mean return")
        ax.set_xlabel("Method")
        ax.set_title(f"{env} — Mean Return")
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
            ax.bar(d["method"], d["mean_return_mean"])
            env_name = str(d["env_id"].iloc[0]) if not d.empty else ""
            ax.set_title(f"{env_name} — Mean Return")
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
    summary: Path = typer.Option(Path("results/summary.csv"), "--summary", "-s", exists=True, dir_okay=False),
    out: Path = typer.Option(Path("results/normalized_scores.png"), "--out", "-o", dir_okay=False),
    highlight: str = typer.Option("proposed", "--highlight"),
) -> None:
    plot_normalized_summary(summary, out, highlight_method=highlight)
    typer.echo(f"[green]Saved normalized bars[/green] to {out}")


def main(argv: Optional[List[str]] = None) -> None:
    app()
