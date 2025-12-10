"""
Generate Figure 1: Comparative Learning Curves across Environments.

This script produces a 2x3 grid of learning curves comparing the Proposed method
against RIDE, RND, ICM, and Vanilla baselines on the core environments.

Usage
-----
python -m scripts.make_fig1 --runs-root runs_suite --out paper_figures/fig1_learning_curves.png
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

from irl.plot import aggregate_runs, expand_run_dirs, AggregateResult

app = typer.Typer(add_completion=False)

# --- Configuration -----------------------------------------------------------

ENV_ORDER = [
    "MountainCar-v0",
    "CarRacing-v3",
    "BipedalWalker-v3",
    "Ant-v5",
    "HalfCheetah-v5",
    "Humanoid-v5",
]

# Map env_id to a cleaner title for the plot
ENV_TITLES = {
    "MountainCar-v0": "MountainCar",
    "CarRacing-v3": "CarRacing",
    "BipedalWalker-v3": "BipedalWalker",
    "Ant-v5": "Ant",
    "HalfCheetah-v5": "HalfCheetah",
    "Humanoid-v5": "Humanoid",
}

# Method style configuration: Color, Line Style, Z-Order (higher = on top)
METHOD_STYLES = {
    "proposed": {"color": "#d62728", "label": "Proposed", "zorder": 10, "lw": 2.0},  # Red
    "ride":     {"color": "#1f77b4", "label": "RIDE",     "zorder": 5, "lw": 1.5},   # Blue
    "rnd":      {"color": "#ff7f0e", "label": "RND",      "zorder": 4, "lw": 1.5},   # Orange
    "icm":      {"color": "#2ca02c", "label": "ICM",      "zorder": 3, "lw": 1.5},   # Green
    "riac":     {"color": "#9467bd", "label": "RIAC",     "zorder": 3, "lw": 1.5},   # Purple
    "vanilla":  {"color": "#7f7f7f", "label": "PPO",      "zorder": 1, "lw": 1.5, "ls": "--"}, # Grey dashed
}

# -----------------------------------------------------------------------------

def _find_runs_for(runs_root: Path, env: str, method: str) -> List[Path]:
    """Find run directories matching env and method in runs_root."""
    # Pattern matches: runs_root / method__env__*
    # Note: env names in dirs have '/' replaced by '-'
    env_tag = env.replace("/", "-")
    pattern = str(runs_root / f"{method}__{env_tag}__*")
    return expand_run_dirs([pattern])


def plot_env_ax(ax, runs_root: Path, env_id: str, smooth: int, shade: bool):
    """Plot all methods for a single environment on a given Axes."""
    env_title = ENV_TITLES.get(env_id, env_id)
    
    # Iterate in z-order so the legend order matches visually, or define fixed order.
    # We'll iterate by the METHOD_STYLES keys to ensure consistent set.
    for method, style in METHOD_STYLES.items():
        runs = _find_runs_for(runs_root, env_id, method)
        if not runs:
            continue

        try:
            agg = aggregate_runs(runs, metric="reward_total_mean", smooth=smooth)
        except Exception:
            # Metric might be missing or empty
            continue

        # Plot mean line
        ax.plot(
            agg.steps, 
            agg.mean, 
            label=style["label"], 
            color=style["color"], 
            linewidth=style.get("lw", 1.5), 
            linestyle=style.get("ls", "-"),
            zorder=style["zorder"]
        )

        # Shade error band
        if shade and agg.n_runs >= 2 and agg.std.size > 0:
            lo = agg.mean - agg.std
            hi = agg.mean + agg.std
            ax.fill_between(
                agg.steps, lo, hi, 
                color=style["color"], 
                alpha=0.15, 
                linewidth=0,
                zorder=style["zorder"] - 0.5 # slightly behind line
            )

    ax.set_title(env_title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    # Formatting
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    # Remove top/right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


@app.command()
def main(
    runs_root: Path = typer.Option(..., help="Path to the suite runs directory."),
    out: Path = typer.Option(Path("paper_figures/fig1.png"), help="Output path."),
    smooth: int = typer.Option(10, help="Smoothing window."),
    shade: bool = typer.Option(True, help="Show error bands."),
):
    """Generate Figure 1: Comparative learning curves."""
    # Setup grid: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, env_id in zip(axes_flat, ENV_ORDER):
        plot_env_ax(ax, runs_root, env_id, smooth, shade)

    # Common labels
    # Use a big "Environment Steps" label at bottom
    # Use "Average Return" label on left
    
    # Hide x labels for top row, y labels for middle/right cols? 
    # Or keep them for clarity since ranges differ vastly.
    # Let's keep them but maybe simplify.
    
    for i, ax in enumerate(axes_flat):
        if i >= 3: # Bottom row
            ax.set_xlabel("Steps")
        if i % 3 == 0: # Left column
            ax.set_ylabel("Average Return")

    # Global Legend (collected from first ax that has handles)
    # We want a single legend, maybe at the bottom or top.
    handles, labels = axes_flat[0].get_legend_handles_labels()
    # Sort legend by desired order (Proposed first)
    # Helper to map label back to priority
    def label_priority(l):
        # find key in METHOD_STYLES where val['label'] == l
        for k, v in METHOD_STYLES.items():
            if v["label"] == l:
                return -v["zorder"] # negate so high zorder comes first
        return 0
    
    if handles:
        # Sort handles/labels together
        hl = sorted(zip(handles, labels), key=lambda x: label_priority(x[1]))
        h_sorted, l_sorted = zip(*hl)
        
        fig.legend(
            h_sorted, l_sorted, 
            loc="upper center", 
            bbox_to_anchor=(0.5, 0.0), 
            ncol=len(METHOD_STYLES), 
            frameon=False,
            fontsize=12
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved Figure 1 to {out}")

if __name__ == "__main__":
    app()
