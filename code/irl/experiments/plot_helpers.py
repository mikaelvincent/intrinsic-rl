# code/irl/experiments/plot_helpers.py
"""Helper functions for experiment suite plotting.

These helpers were originally defined in :mod:`irl.experiments.plotting` and
are split out to keep the main orchestration module smaller and easier to scan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

# Ensure a non-interactive backend for headless environments before importing pyplot
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402

from irl.plot import (
    _aggregate_runs,
    plot_trajectory_heatmap,
)
from irl.utils.checkpoint import atomic_replace


def _generate_comparison_plot(
    groups_by_env: Dict[str, Dict[str, List[Path]]],
    methods_to_plot: List[str],
    metric: str,
    smooth: int,
    shade: bool,
    title: str,
    filename_suffix: str,
    plots_root: Path,
) -> None:
    """Generate one plot per environment comparing specific methods.

    Designed to highlight the 'Proposed' method by plotting it last (on top)
    and applying thicker lines/higher opacity relative to baselines.
    """
    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        # Filter available methods
        relevant_methods = [m for m in methods_to_plot if m in by_method]
        if not relevant_methods:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        any_plotted = False

        # Iterate methods in the requested order.
        # This determines Z-order: later items are drawn on top.
        for method in relevant_methods:
            dirs = by_method[method]
            try:
                agg = _aggregate_runs(dirs, metric=metric, smooth=int(smooth))
            except Exception:
                continue

            if agg.n_runs == 0:
                continue

            # Visual Emphasis Logic:
            # - Proposed method gets higher z-order, thicker line, full alpha, and explicit red color.
            # - Baselines are more transparent (0.4), thinner, and rely on cycle colors.
            # - "Inflate" effect via visual hierarchy: Proposed stands out sharply against a noisy background.
            is_main_proposed = method.lower() == "proposed"

            lw = 2.5 if is_main_proposed else 1.5
            alpha = 1.0 if is_main_proposed else 0.4  # more transparent baselines
            zorder = 10 if is_main_proposed else 2
            color = "#d62728" if is_main_proposed else None  # tab:red for proposed

            label = f"{method} (n={agg.n_runs})"
            ax.plot(
                agg.steps,
                agg.mean,
                label=label,
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
                color=color,
            )
            any_plotted = True

            if shade and agg.n_runs >= 2 and agg.std.size > 0:
                lo = agg.mean - agg.std
                hi = agg.mean + agg.std
                # Pass color so shading matches line, but with low alpha
                ax.fill_between(
                    agg.steps,
                    lo,
                    hi,
                    alpha=0.15,
                    linewidth=0,
                    zorder=zorder - 1,
                    color=color,
                )

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Environment steps")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{env_id} — {title}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        env_tag = env_id.replace("/", "-")
        out = plots_root / f"{env_tag}__{filename_suffix}.png"
        tmp = out.with_suffix(out.suffix + ".tmp")
        fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format="png")
        atomic_replace(tmp, out)
        plt.close(fig)
        typer.echo(f"[suite] Saved {filename_suffix} plot: {out}")


def _generate_gating_plot(
    groups_by_env: Dict[str, Dict[str, List[Path]]],
    plots_root: Path,
    smooth: int = 25,
) -> None:
    """Generate dual-axis plot: Extrinsic Reward vs Gate Rate for Proposed method.

    Visualizes the correlation between opening/closing regions (Gating) and
    performance breakthroughs.
    """
    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        # We only care about the 'proposed' method for this mechanism plot
        runs = by_method.get("proposed")
        if not runs:
            # Fallback: try case-insensitive lookup
            for m, r in by_method.items():
                if m.lower() == "proposed":
                    runs = r
                    break

        if not runs:
            continue

        # Aggregate Extrinsic Reward
        try:
            agg_rew = _aggregate_runs(runs, metric="reward_mean", smooth=smooth)
            # Try gate_rate (fraction 0-1) first, else gate_rate_pct
            agg_gate = _aggregate_runs(runs, metric="gate_rate", smooth=smooth)
            if agg_gate.n_runs == 0:
                agg_gate = _aggregate_runs(runs, metric="gate_rate_pct", smooth=smooth)
        except Exception:
            continue

        if agg_rew.n_runs == 0 or agg_gate.n_runs == 0:
            continue

        # Plot setup
        fig, ax1 = plt.subplots(figsize=(9, 5))

        # Left Axis: Reward (Blue)
        color1 = "tab:blue"
        ax1.set_xlabel("Environment steps")
        ax1.set_ylabel("Extrinsic Reward", color=color1, fontweight="bold")
        ax1.plot(agg_rew.steps, agg_rew.mean, color=color1, linewidth=2.0, label="Reward")
        ax1.tick_params(axis="y", labelcolor=color1)

        # Shade reward variance
        if agg_rew.n_runs >= 2 and agg_rew.std.size > 0:
            ax1.fill_between(
                agg_rew.steps,
                agg_rew.mean - agg_rew.std,
                agg_rew.mean + agg_rew.std,
                color=color1,
                alpha=0.15,
                linewidth=0,
            )

        # Right Axis: Gate Rate (Red)
        ax2 = ax1.twinx()
        color2 = "tab:red"
        # If mean is small (<1.1), assume fraction 0-1, label as Rate. Else %, label as %.
        is_pct = agg_gate.mean.max() > 1.1
        ylabel = "Gate Rate (%)" if is_pct else "Gate Rate (0-1)"
        if not is_pct:
            # Force 0-1 limits if fraction, for clarity
            ax2.set_ylim(0, 1.05)

        ax2.set_ylabel(ylabel, color=color2, fontweight="bold")
        ax2.plot(
            agg_gate.steps,
            agg_gate.mean,
            color=color2,
            linewidth=2.0,
            linestyle="--",
            label="Gate Rate",
        )
        ax2.tick_params(axis="y", labelcolor=color2)

        # Shade gate variance
        if agg_gate.n_runs >= 2 and agg_gate.std.size > 0:
            ax2.fill_between(
                agg_gate.steps,
                agg_gate.mean - agg_gate.std,
                agg_gate.mean + agg_gate.std,
                color=color2,
                alpha=0.15,
                linewidth=0,
            )

        ax1.set_title(f"{env_id} — Gating Dynamics (Proposed)")
        ax1.grid(True, alpha=0.3)

        env_tag = env_id.replace("/", "-")
        out = plots_root / f"{env_tag}__gating_dynamics.png"

        # Save atomic
        tmp = out.with_suffix(out.suffix + ".tmp")
        fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format="png")
        atomic_replace(tmp, out)
        plt.close(fig)
        typer.echo(f"[suite] Saved gating plot: {out}")


def _generate_component_plot(
    groups_by_env: Dict[str, Dict[str, List[Path]]],
    plots_root: Path,
    smooth: int = 25,
) -> None:
    """Generate multi-line plot: Impact RMS vs LP RMS for Proposed method.

    Visualizes the evolution of intrinsic signal magnitudes (drivers) over time.
    """
    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        # We only care about the 'proposed' method
        runs = by_method.get("proposed")
        if not runs:
            # Fallback case-insensitive
            for m, r in by_method.items():
                if m.lower() == "proposed":
                    runs = r
                    break

        if not runs:
            continue

        # Aggregate RMS metrics (logged via scalars.csv in Proposed)
        try:
            agg_imp = _aggregate_runs(runs, metric="impact_rms", smooth=smooth)
            agg_lp = _aggregate_runs(runs, metric="lp_rms", smooth=smooth)
        except Exception:
            continue

        if agg_imp.n_runs == 0 or agg_lp.n_runs == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))

        # Plot Impact RMS
        ax.plot(
            agg_imp.steps,
            agg_imp.mean,
            label="Impact (Novelty)",
            color="#1f77b4",  # Muted Blue
            linewidth=2.5,
            alpha=0.9,
        )
        if agg_imp.n_runs >= 2 and agg_imp.std.size > 0:
            ax.fill_between(
                agg_imp.steps,
                agg_imp.mean - agg_imp.std,
                agg_imp.mean + agg_imp.std,
                color="#1f77b4",
                alpha=0.15,
                linewidth=0,
            )

        # Plot LP RMS
        ax.plot(
            agg_lp.steps,
            agg_lp.mean,
            label="LP (Competence)",
            color="#ff7f0e",  # Safety Orange
            linewidth=2.5,
            alpha=0.9,
        )
        if agg_lp.n_runs >= 2 and agg_lp.std.size > 0:
            ax.fill_between(
                agg_lp.steps,
                agg_lp.mean - agg_lp.std,
                agg_lp.mean + agg_lp.std,
                color="#ff7f0e",
                alpha=0.15,
                linewidth=0,
            )

        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Running RMS (Signal Magnitude)")
        ax.set_title(f"{env_id} — Intrinsic Component Evolution (Proposed)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        env_tag = env_id.replace("/", "-")
        out = plots_root / f"{env_tag}__component_evolution.png"

        tmp = out.with_suffix(out.suffix + ".tmp")
        fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format="png")
        atomic_replace(tmp, out)
        plt.close(fig)
        typer.echo(f"[suite] Saved component plot: {out}")


def _generate_trajectory_plots(
    results_dir: Path,
    plots_root: Path,
) -> None:
    """Generate heatmaps for all saved trajectories in results_dir/plots/trajectories.

    Looks for .npz files saved by the evaluation step.
    """
    traj_dir = results_dir / "plots" / "trajectories"
    if not traj_dir.exists():
        return

    npz_files = list(traj_dir.glob("*_trajectory.npz"))
    if not npz_files:
        return

    typer.echo(f"[suite] Generating trajectory heatmaps for {len(npz_files)} files...")

    for npz_file in npz_files:
        # File name convention: {env_id}_trajectory.npz
        env_tag = npz_file.stem.replace("_trajectory", "")
        out_name = f"{env_tag}__state_heatmap.png"
        out_path = plots_root / out_name

        try:
            plot_trajectory_heatmap(npz_file, out_path)
            typer.echo(f"[suite] Saved heatmap: {out_path}")
        except Exception as exc:
            typer.echo(f"[warn] Failed to plot heatmap for {npz_file}: {exc}")
