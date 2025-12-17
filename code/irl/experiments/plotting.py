from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer

from irl.plot import _parse_run_name, plot_normalized_summary
from irl.utils.checkpoint import load_checkpoint
from irl.utils.runs import discover_runs_by_logs, find_latest_ckpt
from irl.visualization.data import _read_scalars
from irl.visualization.suite_figures import (
    _generate_comparison_plot,
    _generate_component_plot,
    _generate_gating_plot,
    _generate_trajectory_plots,
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


def run_plots_suite(
    runs_root: Path,
    results_dir: Path,
    metric: Optional[str],
    smooth: int,
    shade: bool,
) -> None:
    root = runs_root.resolve()
    results_root = results_dir.resolve()

    _list_dirs("Runs root", root)
    _list_dirs("Results dir", results_root)

    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

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
            df = _read_scalars(rd)
        except Exception as exc:
            skipped_runs[rd] = f"Unreadable logs/scalars.csv ({type(exc).__name__}: {exc})"
            continue

        if df.empty:
            skipped_runs[rd] = "logs/scalars.csv has no data rows"
            continue

        info = _parse_run_name(rd)
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

    plots_root = (results_root / "plots").resolve()
    plots_root.mkdir(parents=True, exist_ok=True)

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
        )
    else:
        preferred = ["vanilla", "icm", "rnd", "ride", "riac"]
        baselines: list[str] = [m for m in preferred if m in all_methods]
        extras = [
            m
            for m in all_methods
            if m not in baselines and m != "glpe" and not m.startswith("glpe_")
        ]
        baselines.extend(extras)
        if "glpe" in all_methods:
            baselines.append("glpe")

        ablation_priority = ["glpe_lp_only", "glpe_impact_only", "glpe_nogate"]
        ablations: list[str] = [m for m in ablation_priority if m in all_methods]
        other_abls = sorted([m for m in all_methods if m.startswith("glpe_") and m not in ablations])
        ablations.extend(other_abls)
        if "glpe" in all_methods:
            ablations.append("glpe")

        _generate_comparison_plot(
            groups,
            methods_to_plot=baselines,
            metric="episode_return_mean",
            smooth=15,
            shade=True,
            title="Task Performance (Episode Return)",
            filename_suffix="perf_extrinsic",
            plots_root=plots_root,
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
        )

        _generate_gating_plot(groups, plots_root=plots_root, smooth=25)
        _generate_component_plot(groups, plots_root=plots_root, smooth=25)

        summary_csv = results_root / "summary.csv"
        if summary_csv.exists():
            bar_plot_path = plots_root / "summary_normalized_bars.png"
            plot_normalized_summary(summary_csv, bar_plot_path, highlight_method="glpe")
            typer.echo(f"[suite] Saved normalized summary bars: {bar_plot_path}")
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
