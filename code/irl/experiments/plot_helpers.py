from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

from irl.plot import _aggregate_runs, plot_trajectory_heatmap
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
    _ = shade

    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        relevant_methods = [m for m in methods_to_plot if m in by_method]
        if not relevant_methods:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        any_plotted = False

        for method in relevant_methods:
            dirs = by_method[method]
            try:
                agg = _aggregate_runs(dirs, metric=metric, smooth=int(smooth))
            except Exception:
                continue

            if agg.n_runs == 0:
                continue

            is_main_proposed = method.lower() == "proposed"
            lw = 2.5 if is_main_proposed else 1.5
            alpha = 1.0 if is_main_proposed else 0.4
            zorder = 10 if is_main_proposed else 2
            color = "#d62728" if is_main_proposed else None

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
    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        runs = by_method.get("proposed")
        if not runs:
            for m, r in by_method.items():
                if m.lower() == "proposed":
                    runs = r
                    break
        if not runs:
            continue

        try:
            agg_rew = _aggregate_runs(runs, metric="reward_mean", smooth=smooth)
            agg_gate = _aggregate_runs(runs, metric="gate_rate", smooth=smooth)
            if agg_gate.n_runs == 0:
                agg_gate = _aggregate_runs(runs, metric="gate_rate_pct", smooth=smooth)
        except Exception:
            continue

        if agg_rew.n_runs == 0 or agg_gate.n_runs == 0:
            continue

        fig, ax1 = plt.subplots(figsize=(9, 5))

        color1 = "tab:blue"
        ax1.set_xlabel("Environment steps")
        ax1.set_ylabel("Extrinsic Reward", color=color1, fontweight="bold")
        ax1.plot(agg_rew.steps, agg_rew.mean, color=color1, linewidth=2.0, label="Reward")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "tab:red"
        is_pct = agg_gate.mean.max() > 1.1
        ylabel = "Gate Rate (%)" if is_pct else "Gate Rate (0-1)"
        if not is_pct:
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

        ax1.set_title(f"{env_id} — Gating Dynamics (Proposed)")
        ax1.grid(True, alpha=0.3)

        env_tag = env_id.replace("/", "-")
        out = plots_root / f"{env_tag}__gating_dynamics.png"
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
    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        runs = by_method.get("proposed")
        if not runs:
            for m, r in by_method.items():
                if m.lower() == "proposed":
                    runs = r
                    break
        if not runs:
            continue

        try:
            agg_imp = _aggregate_runs(runs, metric="impact_rms", smooth=smooth)
            agg_lp = _aggregate_runs(runs, metric="lp_rms", smooth=smooth)
        except Exception:
            continue

        if agg_imp.n_runs == 0 or agg_lp.n_runs == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            agg_imp.steps,
            agg_imp.mean,
            label="Impact (Novelty)",
            color="#1f77b4",
            linewidth=2.5,
            alpha=0.9,
        )
        ax.plot(
            agg_lp.steps,
            agg_lp.mean,
            label="LP (Competence)",
            color="#ff7f0e",
            linewidth=2.5,
            alpha=0.9,
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


def _generate_trajectory_plots(results_dir: Path, plots_root: Path) -> None:
    traj_dir = results_dir / "plots" / "trajectories"
    if not traj_dir.exists():
        return

    npz_files = list(traj_dir.glob("*_trajectory.npz"))
    if not npz_files:
        return

    typer.echo(f"[suite] Generating trajectory heatmaps for {len(npz_files)} files...")

    for npz_file in npz_files:
        env_tag = npz_file.stem.replace("_trajectory", "")
        out_path = plots_root / f"{env_tag}__state_heatmap.png"
        try:
            plot_trajectory_heatmap(npz_file, out_path)
            typer.echo(f"[suite] Saved heatmap: {out_path}")
        except Exception as exc:
            typer.echo(f"[warn] Failed to plot heatmap for {npz_file}: {exc}")
