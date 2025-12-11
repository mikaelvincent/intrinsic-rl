# code/irl/experiments/plotting.py
"""Plotting helpers for the experiment suite.

If ``metric`` is provided, generates one generic overlay plot per environment
including all methods found.

If ``metric`` is ``None`` (Paper Mode), generates specific comparative plots:
  1. Main Comparison (Extrinsic): Proposed vs Baselines (reward_mean).
  2. Main Comparison (Total): Proposed vs Baselines (reward_total_mean).
  3. Ablation Study: Proposed vs Variants (reward_mean & reward_total_mean).
  4. Gating Dynamics (Extrinsic vs Gate Rate).
  5. Intrinsic Component Evolution (Impact vs LP RMS).
  6. Normalized Performance Profile (Bar Chart).
  7. Trajectory Heatmaps (State Space).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import typer  # noqa: E402

from irl.plot import (
    _aggregate_runs,
    _parse_run_name,
    plot_normalized_summary,
)
from .plot_helpers import (
    _generate_comparison_plot,
    _generate_gating_plot,
    _generate_component_plot,
    _generate_trajectory_plots,
)


def run_plots_suite(
    runs_root: Path,
    results_dir: Path,
    metric: Optional[str],
    smooth: int,
    shade: bool,
) -> None:
    """Generate per-environment overlay plots.

    If ``metric`` is provided, generates one generic overlay plot per environment
    including all methods found.

    If ``metric`` is ``None`` (Paper Mode), generates specific comparative plots:
      1. Main Comparison (Extrinsic): Proposed vs Baselines (reward_mean).
      2. Main Comparison (Total): Proposed vs Baselines (reward_total_mean).
      3. Ablation Study: Proposed vs Variants (reward_mean & reward_total_mean).
      4. Gating Dynamics (Extrinsic vs Gate Rate).
      5. Intrinsic Component Evolution (Impact vs LP RMS).
      6. Normalized Performance Profile (Bar Chart).
      7. Trajectory Heatmaps (State Space).
    """
    root = runs_root.resolve()
    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not run_dirs:
        typer.echo(f"[suite] No run directories under {root}")
        return

    # Group run dirs by env and method using the same parser as irl.plot
    groups: Dict[str, Dict[str, List[Path]]] = {}
    for rd in run_dirs:
        info = _parse_run_name(rd)
        env = info.get("env")
        method = info.get("method")
        if not env or not method:
            continue
        groups.setdefault(env, {}).setdefault(method, []).append(rd)

    if not groups:
        typer.echo("[suite] No env/method information could be parsed from run directories.")
        return

    plots_root = (results_dir / "plots").resolve()
    plots_root.mkdir(parents=True, exist_ok=True)

    if metric is not None:
        # Legacy/Single-metric mode: Plot everything found
        _generate_comparison_plot(
            groups,
            methods_to_plot=sorted({m for m_map in groups.values() for m in m_map}),
            metric=metric,
            smooth=smooth,
            shade=shade,
            title=f"All Methods ({metric})",
            filename_suffix=f"overlay_{metric}",
            plots_root=plots_root,
        )
        return

    # --- Paper Mode: Generate specific figures ---

    # Define method order for plotting. Last item is drawn on top.
    # We deliberately place 'proposed' last to ensure its line and shading
    # overlay the baselines for maximum visibility.
    baselines = ["vanilla", "icm", "rnd", "ride", "proposed"]

    # Ablations: Strictly use the components requested by the design plan.
    # Order: Weakest first -> Strongest (Proposed) last.
    ablations = [
        "proposed_lp_only",
        "proposed_impact_only",
        "proposed_nogate",
        "proposed",
    ]

    # 1. Main Comparison (Extrinsic)
    _generate_comparison_plot(
        groups,
        methods_to_plot=baselines,
        metric="reward_mean",
        smooth=15,
        shade=True,
        title="Task Performance (Extrinsic Reward)",
        filename_suffix="perf_extrinsic",
        plots_root=plots_root,
    )

    # 2. Main Comparison (Total Reward)
    _generate_comparison_plot(
        groups,
        methods_to_plot=baselines,
        metric="reward_total_mean",
        smooth=25,
        shade=True,
        title="Total Reward Objective (Smoothed)",
        filename_suffix="perf_total",
        plots_root=plots_root,
    )

    # 3. Ablation Study (Extrinsic)
    _generate_comparison_plot(
        groups,
        methods_to_plot=ablations,
        metric="reward_mean",
        smooth=15,
        shade=True,
        title="Ablation Study (Extrinsic Reward)",
        filename_suffix="ablations",
        plots_root=plots_root,
    )

    # 3b. Ablation Study (Total)
    _generate_comparison_plot(
        groups,
        methods_to_plot=ablations,
        metric="reward_total_mean",
        smooth=25,
        shade=True,
        title="Ablation Study (Total Reward Objective)",
        filename_suffix="ablations_total",
        plots_root=plots_root,
    )

    # 4. Gating Dynamics Plot (Dual Axis)
    _generate_gating_plot(
        groups,
        plots_root=plots_root,
        smooth=25,
    )

    # 5. Intrinsic Component Evolution
    _generate_component_plot(
        groups,
        plots_root=plots_root,
        smooth=25,
    )

    # 6. Normalized Performance Summary (Bar Chart)
    summary_csv = results_dir / "summary.csv"
    if summary_csv.exists():
        bar_plot_path = plots_root / "summary_normalized_bars.png"
        plot_normalized_summary(summary_csv, bar_plot_path, highlight_method="proposed")
        typer.echo(f"[suite] Saved normalized summary bars: {bar_plot_path}")
    else:
        typer.echo("[suite] Skipping bar chart (summary.csv not found; run 'eval' stage first).")

    # 7. Trajectory Heatmaps
    _generate_trajectory_plots(results_dir, plots_root)
