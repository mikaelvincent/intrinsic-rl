from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

from irl.utils.checkpoint import atomic_replace
from irl.visualization.data import aggregate_runs
from irl.visualization.figures import plot_trajectory_heatmap


def _meta_tags(*, smooth: int, align: str) -> tuple[str, str, str]:
    a = str(align).strip().lower() or "interpolate"
    title_tag = f"smooth={int(smooth)}, align={a}"
    file_tag = f"smooth{int(smooth)}__align{a}"
    return a, title_tag, file_tag


def _generate_comparison_plot(
    groups_by_env: Dict[str, Dict[str, List[Path]]],
    methods_to_plot: List[str],
    metric: str,
    smooth: int,
    shade: bool,
    title: str,
    filename_suffix: str,
    plots_root: Path,
    *,
    paper_mode: bool = False,
    align: str = "interpolate",
) -> None:
    align_mode, meta_title, meta_file = _meta_tags(smooth=int(smooth), align=str(align))
    neutral = bool(paper_mode)

    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        relevant_methods = [m for m in methods_to_plot if m in by_method]
        if not relevant_methods:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        any_plotted = False

        for method in relevant_methods:
            dirs = by_method[method]
            try:
                agg = aggregate_runs(dirs, metric=metric, smooth=int(smooth), align=align_mode)
            except Exception:
                continue

            if agg.n_runs == 0 or agg.steps.size == 0:
                continue

            is_main_glpe = (method.lower() == "glpe") and not neutral
            lw = 2.5 if is_main_glpe else 1.5
            alpha = 1.0 if is_main_glpe else 0.4
            zorder = 10 if is_main_glpe else 2
            color = "#d62728" if is_main_glpe else None

            if neutral:
                lw = 1.8
                alpha = 0.9
                zorder = 2
                color = None

            label = f"{method} (n={agg.n_runs})"
            line = ax.plot(
                agg.steps,
                agg.mean,
                label=label,
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
                color=color,
            )[0]

            if shade and agg.n_runs > 1:
                ci = 1.96 * (agg.std / sqrt(float(agg.n_runs)))
                shade_alpha = 0.12 if neutral else (0.18 if is_main_glpe else 0.12)
                ax.fill_between(
                    agg.steps,
                    agg.mean - ci,
                    agg.mean + ci,
                    alpha=shade_alpha,
                    color=line.get_color(),
                    linewidth=0.0,
                    zorder=max(0, zorder - 1),
                )

            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Environment steps")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{env_id} — {title} ({meta_title})")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        env_tag = env_id.replace("/", "-")
        out = plots_root / f"{env_tag}__{filename_suffix}__{meta_file}.png"
        tmp = out.with_suffix(out.suffix + ".tmp")
        fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format="png")
        atomic_replace(tmp, out)
        plt.close(fig)
        typer.echo(f"[suite] Saved {filename_suffix} plot: {out}")


def _generate_gating_plot(
    groups_by_env: Dict[str, Dict[str, List[Path]]],
    plots_root: Path,
    smooth: int = 25,
    *,
    align: str = "interpolate",
) -> None:
    align_mode, meta_title, meta_file = _meta_tags(smooth=int(smooth), align=str(align))

    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        runs = by_method.get("glpe")
        if not runs:
            for m, r in by_method.items():
                if m.lower() == "glpe":
                    runs = r
                    break
        if not runs:
            continue

        try:
            agg_rew = aggregate_runs(runs, metric="reward_mean", smooth=smooth, align=align_mode)
            agg_gate = aggregate_runs(runs, metric="gate_rate", smooth=smooth, align=align_mode)
            if agg_gate.n_runs == 0:
                agg_gate = aggregate_runs(
                    runs, metric="gate_rate_pct", smooth=smooth, align=align_mode
                )
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

        ax1.set_title(f"{env_id} — Gating Dynamics (GLPE) ({meta_title})")
        ax1.grid(True, alpha=0.3)

        env_tag = env_id.replace("/", "-")
        out = plots_root / f"{env_tag}__gating_dynamics__{meta_file}.png"
        tmp = out.with_suffix(out.suffix + ".tmp")
        fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format="png")
        atomic_replace(tmp, out)
        plt.close(fig)
        typer.echo(f"[suite] Saved gating plot: {out}")


def _generate_component_plot(
    groups_by_env: Dict[str, Dict[str, List[Path]]],
    plots_root: Path,
    smooth: int = 25,
    *,
    align: str = "interpolate",
) -> None:
    align_mode, meta_title, meta_file = _meta_tags(smooth=int(smooth), align=str(align))

    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: kv[0]):
        runs = by_method.get("glpe")
        if not runs:
            for m, r in by_method.items():
                if m.lower() == "glpe":
                    runs = r
                    break
        if not runs:
            continue

        try:
            agg_imp = aggregate_runs(runs, metric="impact_rms", smooth=smooth, align=align_mode)
            agg_lp = aggregate_runs(runs, metric="lp_rms", smooth=smooth, align=align_mode)
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
        ax.set_title(f"{env_id} — Intrinsic Component Evolution (GLPE) ({meta_title})")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        env_tag = env_id.replace("/", "-")
        out = plots_root / f"{env_tag}__component_evolution__{meta_file}.png"
        tmp = out.with_suffix(out.suffix + ".tmp")
        fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format="png")
        atomic_replace(tmp, out)
        plt.close(fig)
        typer.echo(f"[suite] Saved component plot: {out}")


def _generate_trajectory_plots(results_dir: Path, _plots_root: Path) -> None:
    traj_dir = results_dir / "plots" / "trajectories"
    if not traj_dir.exists():
        return

    npz_files = sorted(traj_dir.rglob("*_trajectory.npz"), key=lambda p: str(p))
    if not npz_files:
        return

    typer.echo(f"[suite] Generating trajectory heatmaps for {len(npz_files)} files...")

    for npz_file in npz_files:
        env_tag = npz_file.stem.replace("_trajectory", "")
        out_path = npz_file.with_name(f"{env_tag}__state_heatmap.png")
        try:
            wrote = bool(plot_trajectory_heatmap(npz_file, out_path))
            if wrote:
                typer.echo(f"[suite] Saved heatmap: {out_path}")
            else:
                typer.echo(f"[suite] Skipped heatmap (no projection): {npz_file}")
        except Exception as exc:
            typer.echo(f"[warn] Failed to plot heatmap for {npz_file}: {exc}")
