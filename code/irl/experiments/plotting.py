from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import typer

from irl.utils.runs import discover_runs_by_logs, parse_run_name


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _list_dirs(label: str, root: Path) -> List[Path]:
    if not root.exists():
        typer.echo(f"[suite] {label}: {root} (missing)")
        return []
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    typer.echo(f"[suite] {label}: {root} ({len(dirs)} directorie(s))")
    for d in dirs:
        typer.echo(f"[suite]   - {d.name}")
    return dirs


def _groups_for_timing(root: Path) -> Dict[str, Dict[str, List[Path]]]:
    groups: Dict[str, Dict[str, List[Path]]] = {}
    for rd in discover_runs_by_logs(root):
        info = parse_run_name(rd)
        method = (str(info.get("method") or "")).strip().lower() or "unknown"
        env = (str(info.get("env") or "")).strip() or "unknown_env"
        env = env.replace("/", "-")
        groups.setdefault(env, {}).setdefault(method, []).append(rd)
    return groups


def run_plots_suite(
    *,
    runs_root: Path,
    results_dir: Path,
) -> None:
    root = runs_root.resolve()
    results_root = results_dir.resolve()

    _list_dirs("Runs root", root)
    _list_dirs("Results dir", results_root)

    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    plots_root = (results_root / "plots").resolve()
    plots_root.mkdir(parents=True, exist_ok=True)

    summary_csv = results_root / "summary.csv"
    if not summary_csv.exists():
        typer.echo("[suite] Skipping plots (summary.csv not found; run 'eval' stage first).")
        return

    try:
        from irl.visualization.paper_figures import (
            load_eval_by_step_table,
            load_eval_summary_table,
            paper_method_groups,
            plot_eval_auc_bars_by_env,
            plot_eval_bars_by_env,
            plot_eval_curves_by_env,
            plot_glpe_extrinsic_vs_intrinsic,
            plot_glpe_state_gate_map,
            plot_steps_to_beat_by_env,
        )

        df_summary = load_eval_summary_table(summary_csv)
        methods_present = sorted(set(df_summary["method_key"].tolist()))
        baselines, ablations = paper_method_groups(methods_present)

        plot_eval_bars_by_env(
            df_summary,
            plots_root=plots_root,
            methods_to_plot=baselines,
            title="Task performance (evaluation)",
            filename_suffix="paper_baselines",
        )
        plot_eval_bars_by_env(
            df_summary,
            plots_root=plots_root,
            methods_to_plot=ablations,
            title="Ablation study (evaluation)",
            filename_suffix="paper_ablations",
        )

        by_step_path = results_root / "summary_by_step.csv"
        if by_step_path.exists():
            df_steps = load_eval_by_step_table(by_step_path)
            plot_eval_curves_by_env(
                df_steps,
                plots_root=plots_root,
                methods_to_plot=baselines,
                title="Learning curves (evaluation)",
                filename_suffix="paper_baselines_curves",
            )
            plot_eval_curves_by_env(
                df_steps,
                plots_root=plots_root,
                methods_to_plot=ablations,
                title="Ablations over training (evaluation)",
                filename_suffix="paper_ablations_curves",
            )

            plot_eval_auc_bars_by_env(
                df_steps,
                plots_root=plots_root,
                methods_to_plot=baselines,
                title="Sample efficiency (AUC of eval return)",
                filename_suffix="paper_baselines",
            )
            plot_eval_auc_bars_by_env(
                df_steps,
                plots_root=plots_root,
                methods_to_plot=ablations,
                title="Sample efficiency (AUC of eval return)",
                filename_suffix="paper_ablations",
            )

            raw_path = results_root / "summary_raw.csv"

            plot_steps_to_beat_by_env(
                df_steps,
                plots_root=plots_root,
                methods_to_plot=baselines,
                title="Steps to beat score threshold (GLPE vs baselines)",
                filename_suffix="paper_baselines",
                summary_raw_csv=raw_path if raw_path.exists() else None,
            )
            plot_steps_to_beat_by_env(
                df_steps,
                plots_root=plots_root,
                methods_to_plot=ablations,
                title="Steps to beat score threshold (GLPE ablations)",
                filename_suffix="paper_ablations",
                summary_raw_csv=raw_path if raw_path.exists() else None,
            )
        else:
            typer.echo("[suite] summary_by_step.csv not found; skipping eval learning curves.")

        traj_root = results_root / "plots" / "trajectories"
        if traj_root.exists():
            plot_glpe_state_gate_map(traj_root=traj_root, plots_root=plots_root)
            plot_glpe_extrinsic_vs_intrinsic(traj_root=traj_root, plots_root=plots_root)
        else:
            typer.echo("[suite] No trajectories found; skipping GLPE trajectory plots.")

        timing_groups = _groups_for_timing(root)

        try:
            from irl.visualization.paper.training_plots import plot_training_reward_decomposition

            written = plot_training_reward_decomposition(timing_groups, plots_root=plots_root)
            if not written:
                typer.echo("[suite] Training reward plots skipped (no data).")
        except Exception as exc:
            typer.echo(f"[suite] Training reward plots skipped ({type(exc).__name__}: {exc})")

        try:
            from irl.visualization.timing_figures import plot_timing_breakdown

            plot_timing_breakdown(timing_groups, plots_root=plots_root, tail_frac=0.25)
        except Exception as exc:
            typer.echo(f"[suite] Timing plots skipped ({type(exc).__name__}: {exc})")

        try:
            from irl.visualization.timing_figures import plot_intrinsic_taper_weight

            taper_written = plot_intrinsic_taper_weight(timing_groups, plots_root=plots_root)
            if not taper_written:
                typer.echo("[suite] Intrinsic taper plot skipped (no taper active).")
        except Exception as exc:
            typer.echo(f"[suite] Intrinsic taper plot skipped ({type(exc).__name__}: {exc})")

    except Exception as exc:
        typer.echo(f"[suite] Plotting failed ({type(exc).__name__}: {exc})")
        return

    typer.echo(f"[suite] Plots written under: {_rel(plots_root, results_root)}")
