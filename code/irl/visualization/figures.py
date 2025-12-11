"""Figure-level plotting helpers for learning curves and summaries.

This module hosts pure plotting utilities that take pre-aggregated data
or trajectory dumps and render Matplotlib figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Ensure a non-interactive backend for headless environments before importing pyplot
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from irl.utils.checkpoint import atomic_replace  # <-- atomic helper
from .data import _ensure_parent


def plot_normalized_summary(
    summary_path: Path,
    out_path: Path,
    highlight_method: str = "proposed",
) -> None:
    """Generate a normalized grouped bar chart (Min-Max scaled per Env).

    This visualization isolates relative performance by scaling the scores
    in each environment to [0, 1], where 0 is the worst method's score and
    1 is the best. This effectively highlights the "Proposed" method's
    dominance (assuming it performs well) while abstracting away raw scale
    differences between environments.
    """
    if not summary_path.exists():
        return

    df = pd.read_csv(summary_path)
    required = {"method", "env_id", "mean_return_mean"}
    if not required.issubset(df.columns):
        return

    # Pivot to [Env, Method] -> Score matrix
    # If multiple entries exist per (env, method), take the mean (robustness)
    pivoted = df.pivot_table(
        index="env_id", columns="method", values="mean_return_mean", aggfunc="mean"
    )

    # Min-Max Normalization per environment (row-wise)
    # (x - min) / (max - min)
    # This forces the range to [0, 1] for every environment.
    mins = pivoted.min(axis=1)
    maxs = pivoted.max(axis=1)
    ranges = maxs - mins
    # Avoid division by zero if all methods have identical scores
    ranges[ranges == 0] = 1.0

    normalized = pivoted.sub(mins, axis=0).div(ranges, axis=0)

    # Ordering: Put highlight_method last (right-most bar) for visual emphasis
    methods = sorted(list(normalized.columns))
    if highlight_method in methods:
        methods.remove(highlight_method)
        methods.append(highlight_method)

    normalized = normalized[methods]

    # Plotting
    n_envs = len(normalized.index)
    n_methods = len(methods)

    if n_envs == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, n_envs * 2), 6))

    x = np.arange(n_envs)
    width = 0.8 / n_methods

    # Define colors: Highlight proposed with a distinct color, others muted
    cmap = plt.get_cmap("tab10")
    colors = []
    for m in methods:
        if m.lower() == highlight_method.lower():
            colors.append("#d62728")  # tab:red (distinct/bold)
        else:
            idx = methods.index(m)
            colors.append(cmap(idx % 10))

    # Draw bars
    for i, method in enumerate(methods):
        vals = normalized[method].values
        vals = np.nan_to_num(vals)

        offset = (i - n_methods / 2) * width + width / 2
        ax.bar(
            x + offset,
            vals,
            width,
            label=method,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9 if method.lower() == highlight_method.lower() else 0.7,
        )

    ax.set_ylabel("Normalized Score (Min-Max Scaled)")
    ax.set_title("Performance Profile: Normalized Extrinsic Return per Environment")
    ax.set_xticks(x)
    ax.set_xticklabels(normalized.index, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Method")

    fig.tight_layout()

    _ensure_parent(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    # Explicitly tell matplotlib the format to avoid "tmp" extension error
    fmt = out_path.suffix.lstrip(".").lower() or "png"
    fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format=fmt)
    atomic_replace(tmp, out_path)
    plt.close(fig)


def plot_trajectory_heatmap(
    npz_path: Path,
    out_path: Path,
    max_points: int = 20000,
) -> None:
    """Generate heatmaps for all saved trajectories in results_dir/plots/trajectories.

    Looks for .npz files saved by the evaluation step.
    """
    if not npz_path.exists():
        return

    try:
        data = np.load(npz_path)
        obs = data["obs"]
        gates = data["gates"]
        # intrinsic = data["intrinsic"]  # currently unused, but available for color
    except Exception:
        return

    # Subsample if too large to keep plotting fast and file size small
    N = obs.shape[0]
    if N > max_points:
        idx = np.linspace(0, N - 1, max_points, dtype=int)
        obs = obs[idx]
        gates = gates[idx]

    # Heuristic projection: Dim 0 vs Dim 1 (Works for MountainCar, Ant position-like dims)
    if obs.shape[1] < 2:
        return  # Cannot plot 2D

    x = obs[:, 0]
    y = obs[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Gated points (0) as background, Active points (1) as foreground highlight
    mask_active = gates == 1
    mask_gated = gates == 0

    if mask_gated.any():
        ax.scatter(
            x[mask_gated],
            y[mask_gated],
            c="lightgray",
            s=10,
            alpha=0.5,
            label="Gated (Mastered/Noise)",
            edgecolor="none",
        )

    if mask_active.any():
        ax.scatter(
            x[mask_active],
            y[mask_active],
            c="tab:red",
            s=15,
            alpha=0.8,
            label="Active (Learning)",
            edgecolor="none",
        )

    ax.set_xlabel("State Dim 0")
    ax.set_ylabel("State Dim 1")
    ax.set_title("Exploration Heatmap: Active vs Gated Regions")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    _ensure_parent(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    fmt = out_path.suffix.lstrip(".").lower() or "png"
    fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format=fmt)
    atomic_replace(tmp, out_path)
    plt.close(fig)
