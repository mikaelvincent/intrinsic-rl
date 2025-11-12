"""Plotting utilities and CLI for learning curves and summaries.

Includes commands for:
- Learning curves with optional smoothing and shaded standard deviation across seeds.
- Overlays of multiple method groups (each group may include many runs).
- Bar charts from aggregated sweep results (``results/summary.csv``).

Typical usage
-------------
# 1) Aggregate learning curve for one method (multiple run dirs via globs)
python -m irl.plot curves \
  --runs "runs/proposed__BipedalWalker*" \
  --metric reward_total_mean \
  --smooth 5 \
  --shade \
  --out results/walker_proposed_curve.png

# 2) Overlay multiple methods (each --group is a comma-separated list of globs)
python -m irl.plot overlay \
  --group "runs/proposed__BipedalWalker*" \
  --group "runs/ride__BipedalWalker*,runs/rnd__BipedalWalker*" \
  --labels "Proposed" \
  --labels "RIDE+RND (combined)" \
  --metric reward_total_mean \
  --smooth 5 \
  --shade \
  --out results/walker_overlay.png

# 3) Bar chart from sweep aggregation
python -m irl.plot bars \
  --summary results/summary.csv \
  --env BipedalWalker-v3 \
  --out results/bars_walker.png
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Use a non-interactive backend for headless environments
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402

from irl.utils.checkpoint import atomic_replace  # <-- atomic helper

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


# ----------------------------- Utilities ---------------------------------


def _dedup_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            out.append(rp)
            seen.add(rp)
    return out


def _expand_run_dirs(patterns: Sequence[str]) -> list[Path]:
    """Expand glob patterns to run directories that contain logs/scalars.csv.

    Accepts patterns that point either to the run directory itself or directly to the CSV file. Returns unique parent
    run directories.
    """
    dirs: list[Path] = []
    for pat in patterns:
        for hit in glob.glob(pat):
            p = Path(hit)
            if p.is_file() and p.name == "scalars.csv" and p.parent.name == "logs":
                dirs.append(p.parent.parent)
            elif p.is_dir():
                if (p / "logs" / "scalars.csv").exists():
                    dirs.append(p)
    return _dedup_paths(dirs)


def _parse_run_name(run_dir: Path) -> dict[str, str]:
    """Best-effort parser for run directory names produced by default_run_dir().

    Format: <method>__<env>__seed<NUM>__<YYYYmmdd-HHMMSS>
    Returns partial info; missing keys are omitted.
    """
    info: dict[str, str] = {}
    name = run_dir.name
    parts = name.split("__")
    if len(parts) >= 1:
        info["method"] = parts[0]
    if len(parts) >= 2:
        info["env"] = parts[1]
    if len(parts) >= 3:
        m = re.match(r"seed(\d+)", parts[2])
        if m:
            info["seed"] = m.group(1)
    return info


def _read_scalars(run_dir: Path) -> pd.DataFrame:
    """Load the scalars CSV for a run; raises if not found."""
    path = run_dir / "logs" / "scalars.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing scalars.csv in {run_dir}")
    df = pd.read_csv(path)
    # Ensure required 'step' column exists and is numeric
    if "step" not in df.columns:
        raise ValueError(f"'step' column not found in {path}")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"])
    df["step"] = df["step"].astype(int)
    return df


def _smooth_series(s: pd.Series, window: int) -> pd.Series:
    """Simple moving average smoothing; window=1 => no smoothing."""
    w = int(max(1, window))
    if w == 1:
        return s
    return s.rolling(window=w, min_periods=1).mean()


@dataclass
class AggregateResult:
    steps: np.ndarray  # (K,)
    mean: np.ndarray  # (K,)
    std: np.ndarray  # (K,)
    n_runs: int  # number of contributing runs
    method_hint: Optional[str]  # from directory name, if consistent
    env_hint: Optional[str]  # from directory name, if consistent


def _aggregate_runs(
    run_dirs: Sequence[Path],
    metric: str,
    smooth: int = 1,
) -> AggregateResult:
    """Aggregate a metric across multiple runs keyed by 'step'.

    Steps from all runs are unioned; at each step we average over the runs that contain that step (no interpolation).
    Smoothing is applied per-run first.
    """
    if not run_dirs:
        raise ValueError("No run directories to aggregate.")

    method_cand: set[str] = set()
    env_cand: set[str] = set()

    series_per_run: list[pd.Series] = []
    for rd in run_dirs:
        info = _parse_run_name(rd)
        if "method" in info:
            method_cand.add(info["method"])
        if "env" in info:
            env_cand.add(info["env"])

        df = _read_scalars(rd)
        if metric not in df.columns:
            # Soft fallback: prefer reward_total_mean, then reward_mean
            fallback = None
            if metric != "reward_total_mean" and "reward_total_mean" in df.columns:
                fallback = "reward_total_mean"
            elif metric != "reward_mean" and "reward_mean" in df.columns:
                fallback = "reward_mean"
            if fallback is None:
                raise ValueError(
                    f"Metric '{metric}' not found in {rd}/logs/scalars.csv; "
                    "no suitable fallback available."
                )
            metric_local = fallback
        else:
            metric_local = metric

        y = pd.to_numeric(df[metric_local], errors="coerce")
        x = df["step"]
        s = pd.Series(y.values, index=x.values).dropna()
        s = _smooth_series(s, smooth)
        series_per_run.append(s)

    # Union all steps and compute mean/std at each step over available runs
    all_steps = sorted(set().union(*[set(s.index) for s in series_per_run]))
    means: list[float] = []
    stds: list[float] = []

    for st in all_steps:
        vals = [float(s[st]) for s in series_per_run if st in s.index]
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0)

    steps_arr = np.asarray(all_steps, dtype=np.int64)
    mean_arr = np.asarray(means, dtype=np.float64)
    std_arr = np.asarray(stds, dtype=np.float64)

    method_hint = list(method_cand)[0] if len(method_cand) == 1 else None
    env_hint = list(env_cand)[0] if len(env_cand) == 1 else None

    return AggregateResult(
        steps=steps_arr,
        mean=mean_arr,
        std=std_arr,
        n_runs=len(series_per_run),
        method_hint=method_hint,
        env_hint=env_hint,
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------- Commands ----------------------------------


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
    fig.savefig(str(tmp), dpi=150, bbox_inches="tight")
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
        '(e.g., --group "runs/proposed__BipedalWalker*" --group "runs/ride__BipedalWalker*,runs/rnd__BipedalWalker*").',
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
            raise typer.BadParameter(f"No runs found for group {i+1} spec: {spec!r}")

        agg = _aggregate_runs(run_dirs, metric=metric, smooth=smooth)
        lbl = (
            (labels[i] if labels and i < len(labels) else None) or agg.method_hint or f"group-{i+1}"
        )
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
    fig.savefig(str(tmp), dpi=150, bbox_inches="tight")
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
        # All envs: plot grouped bars by env (one chart with multiple envs stacked)
        envs = sorted(df["env_id"].unique())
        # Sort each env's slice and optionally keep topK
        rows: list[pd.DataFrame] = []
        for e in envs:
            d = df[df["env_id"] == e].copy()
            d = d.sort_values("mean_return_mean", ascending=False)
            if topk and topk > 0:
                d = d.head(int(topk))
            rows.append(d)

        # Build a long-form table for plotting in panels stacked vertically
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
    fig.savefig(str(tmp), dpi=150, bbox_inches="tight")
    atomic_replace(tmp, out)
    plt.close("all")
    typer.echo(f"[green]Saved[/green] {out}")


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover - CLI entry
    app()


if __name__ == "__main__":
    main()
