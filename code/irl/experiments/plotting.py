from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer

from irl.utils.runs import discover_runs_by_logs, parse_run_name
from .plot_helpers import (
    _generate_comparison_plot,
    _generate_component_plot,
    _generate_gating_plot,
    _generate_trajectory_plots,
    _suite_method_groups,
)


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


def _infer_method_env_from_checkpoint(
    run_dir: Path,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    from irl.utils.checkpoint import load_checkpoint
    from irl.utils.runs import find_latest_ckpt

    ckpt = find_latest_ckpt(run_dir)
    if ckpt is None:
        return None, None, None
    try:
        payload = load_checkpoint(ckpt, map_location="cpu")
    except Exception:
        return None, None, None

    cfg = payload.get("cfg", {}) or {}
    try:
        method = cfg.get("method", None) if isinstance(cfg, dict) else None
    except Exception:
        method = None
    try:
        env_cfg = (cfg.get("env") or {}) if isinstance(cfg, dict) else {}
        env_id = env_cfg.get("id", None) if isinstance(env_cfg, dict) else None
    except Exception:
        env_id = None
    try:
        seed = int(cfg.get("seed", 0)) if isinstance(cfg, dict) else None
    except Exception:
        seed = None

    return (
        str(method) if method is not None else None,
        str(env_id) if env_id is not None else None,
        seed,
    )


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
    runs_root: Path,
    results_dir: Path,
    metric: Optional[str],
    smooth: int,
    shade: bool,
    *,
    paper_mode: bool = False,
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

    if paper_mode:
        summary_csv = results_root / "summary.csv"
        if not summary_csv.exists():
            typer.echo("[suite] Skipping paper plots (summary.csv not found; run 'eval' stage first).")
            return

        try:
            from irl.visualization.paper_figures import (
                load_eval_by_step_table,
                load_eval_summary_table,
                paper_method_groups,
                plot_eval_bars_by_env,
                plot_eval_curves_by_env,
                plot_glpe_extrinsic_vs_intrinsic,
                plot_glpe_state_gate_map,
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
            else:
                typer.echo("[suite] summary_by_step.csv not found; skipping eval learning curves.")

            traj_root = results_root / "plots" / "trajectories"
            if traj_root.exists():
                plot_glpe_state_gate_map(traj_root=traj_root, plots_root=plots_root)
                plot_glpe_extrinsic_vs_intrinsic(traj_root=traj_root, plots_root=plots_root)
            else:
                typer.echo("[suite] No trajectories found; skipping GLPE trajectory plots.")

            try:
                from irl.visualization.timing_figures import plot_timing_breakdown

                timing_groups = _groups_for_timing(root)
                plot_timing_breakdown(timing_groups, plots_root=plots_root, tail_frac=0.25)
            except Exception as exc:
                typer.echo(f"[suite] Timing plots skipped ({type(exc).__name__}: {exc})")

        except Exception as exc:
            typer.echo(f"[suite] Paper plotting failed ({type(exc).__name__}: {exc})")

        return

    from irl.visualization import plot_normalized_summary
    from irl.visualization.data import read_scalars
    from irl.utils.runs import find_latest_ckpt
    from irl.utils.checkpoint import load_checkpoint

    discovered_run_dirs = discover_runs_by_logs(root)
    if not discovered_run_dirs:
        typer.echo(f"[suite] No run directories with logs/scalars.csv under {root}")
        return

    typer.echo(f"[suite] Discovered {len(discovered_run_dirs)} run(s) with logs/scalars.csv")

    groups: Dict[str, Dict[str, List[Path]]] = {}
    processed_runs: list[Path] = []
    skipped_runs: dict[Path, str] = {}

    for rd in discovered_run_dirs:
        try:
            df = read_scalars(rd)
        except Exception as exc:
            skipped_runs[rd] = f"Unreadable logs/scalars.csv ({type(exc).__name__}: {exc})"
            continue

        if df.empty:
            skipped_runs[rd] = "logs/scalars.csv has no data rows"
            continue

        info = parse_run_name(rd)
        method = info.get("method")
        env = info.get("env")

        if env is None or (method is None or method.strip() == ""):
            m2, e2, _ = _infer_method_env_from_checkpoint(rd)
            method = method or m2
            env = env or e2

        method_key = (str(method) if method is not None else "unknown").strip().lower() or "unknown"
        env_key = (str(env) if env is not None else "unknown_env").strip() or "unknown_env"
        env_key = env_key.replace("/", "-")

        groups.setdefault(env_key, {}).setdefault(method_key, []).append(rd)
        processed_runs.append(rd)

    if not groups:
        typer.echo("[suite] No valid runs to plot (all discovered runs were skipped).")
        typer.echo(f"[suite] Skipped {len(skipped_runs)} run directory(ies):")
        for p, reason in sorted(skipped_runs.items(), key=lambda kv: str(kv[0])):
            typer.echo(f"[suite]   - {_rel(p, root)}: {reason}")
        return

    all_methods = sorted({m for m_map in groups.values() for m in m_map})

    if metric is not None:
        _generate_comparison_plot(
            groups,
            methods_to_plot=all_methods,
            metric=metric,
            smooth=smooth,
            shade=shade,
            title=f"All Methods ({metric})",
            filename_suffix=f"overlay_{metric}",
            plots_root=plots_root,
            paper_mode=bool(paper_mode),
        )
    else:
        baselines, ablations = _suite_method_groups(all_methods)

        _generate_comparison_plot(
            groups,
            methods_to_plot=baselines,
            metric="episode_return_mean",
            smooth=15,
            shade=True,
            title="Task Performance (Episode Return)",
            filename_suffix="perf_extrinsic",
            plots_root=plots_root,
            paper_mode=bool(paper_mode),
        )

        _generate_comparison_plot(
            groups,
            methods_to_plot=baselines,
            metric="reward_total_mean",
            smooth=25,
            shade=True,
            title="Total Reward Objective (Smoothed)",
            filename_suffix="perf_total",
            plots_root=plots_root,
            paper_mode=bool(paper_mode),
        )

        _generate_comparison_plot(
            groups,
            methods_to_plot=ablations,
            metric="episode_return_mean",
            smooth=15,
            shade=True,
            title="Ablation Study (Episode Return)",
            filename_suffix="ablations",
            plots_root=plots_root,
            paper_mode=bool(paper_mode),
        )

        _generate_comparison_plot(
            groups,
            methods_to_plot=ablations,
            metric="reward_total_mean",
            smooth=25,
            shade=True,
            title="Ablation Study (Total Reward Objective)",
            filename_suffix="ablations_total",
            plots_root=plots_root,
            paper_mode=bool(paper_mode),
        )

        _generate_gating_plot(groups, plots_root=plots_root, smooth=25)
        _generate_component_plot(groups, plots_root=plots_root, smooth=25)

        try:
            from irl.visualization.timing_figures import plot_timing_breakdown

            plot_timing_breakdown(groups, plots_root=plots_root, tail_frac=0.25)
        except Exception as exc:
            typer.echo(f"[suite] Timing plots skipped ({type(exc).__name__}: {exc})")

        summary_csv = results_root / "summary.csv"
        if summary_csv.exists():
            bar_plot_path = plots_root / "summary_normalized_bars.png"
            plot_normalized_summary(
                summary_csv,
                bar_plot_path,
                highlight_method=None if bool(paper_mode) else "glpe",
                baseline_method="vanilla",
                baseline_required=bool(paper_mode),
            )
            if bar_plot_path.exists():
                typer.echo(f"[suite] Saved normalized summary bars: {bar_plot_path}")
            else:
                typer.echo("[suite] Skipping bar chart (baseline method missing).")
        else:
            typer.echo("[suite] Skipping bar chart (summary.csv not found; run 'eval' stage first).")

        _generate_trajectory_plots(results_root, plots_root)

    typer.echo(f"[suite] Plotting inputs: processed {len(processed_runs)} run(s).")

    top_level_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    top_has_any_run: dict[Path, bool] = {d: False for d in top_level_dirs}
    processed_set = {p.resolve() for p in processed_runs}

    for rd in discovered_run_dirs:
        try:
            rel = rd.relative_to(root)
            if rel.parts:
                top = (root / rel.parts[0]).resolve()
                if top in top_has_any_run:
                    top_has_any_run[top] = True
        except Exception:
            continue

    non_run_dirs: dict[Path, str] = {}
    for d in top_level_dirs:
        if d.resolve() in processed_set:
            continue
        if top_has_any_run.get(d, False):
            non_run_dirs[d] = "Container directory (contains nested run(s))"
        else:
            non_run_dirs[d] = "No logs/scalars.csv found under this directory"

    if skipped_runs or non_run_dirs:
        typer.echo("[suite] Skipped / not processed directories:")
        for p, reason in sorted(non_run_dirs.items(), key=lambda kv: kv[0].name):
            typer.echo(f"[suite]   - {_rel(p, root)}: {reason}")
        for p, reason in sorted(skipped_runs.items(), key=lambda kv: str(kv[0])):
            typer.echo(f"[suite]   - {_rel(p, root)}: {reason}")
    else:
        typer.echo("[suite] No directories were skipped.")
