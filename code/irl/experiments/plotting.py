from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import typer

from irl.utils.checkpoint import atomic_write_text
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


def _iso_utc(epoch_s: float) -> str:
    try:
        return datetime.fromtimestamp(float(epoch_s), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def _safe_mtime(path: Path) -> tuple[float | None, str | None]:
    try:
        mt = float(Path(path).stat().st_mtime)
    except Exception:
        return None, None
    return mt, _iso_utc(mt)


def _dedup_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        pp = Path(p)
        try:
            key = str(pp.resolve())
        except Exception:
            key = str(pp)
        if key in seen:
            continue
        out.append(pp)
        seen.add(key)
    out.sort(key=lambda x: str(x))
    return out


_OBSOLETE_PLOT_PATTERNS: tuple[str, ...] = (
    "*__auc__paper_baselines.png",
    "*__auc__paper_ablations.png",
    "*__glpe_extrinsic_vs_intrinsic.png",
    "*__eval_scatter__paper_all_methods.png",
)


def _remove_obsolete_plots(plots_root: Path) -> list[Path]:
    removed: list[Path] = []
    root = Path(plots_root)
    for pat in _OBSOLETE_PLOT_PATTERNS:
        for p in sorted(root.glob(str(pat)), key=lambda x: x.name):
            try:
                if p.is_file():
                    p.unlink()
                    removed.append(p)
            except Exception:
                continue
    return removed


def _write_plots_manifest(path: Path, payload: dict[str, object]) -> None:
    try:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        atomic_write_text(Path(path), text)
    except Exception as exc:
        typer.echo(f"[suite] WARN  Failed to write plots manifest ({type(exc).__name__}: {exc})")


def run_plots_suite(
    *,
    runs_root: Path,
    results_dir: Path,
) -> None:
    root = Path(runs_root).resolve()
    results_root = Path(results_dir).resolve()

    results_root.mkdir(parents=True, exist_ok=True)
    plots_root = (results_root / "plots").resolve()
    plots_root.mkdir(parents=True, exist_ok=True)
    manifest_path = plots_root / "plots_manifest.json"

    written: list[Path] = []
    notes: list[str] = []
    status: str = "ok"
    reason: str | None = None

    _list_dirs("Runs root", root)
    _list_dirs("Results dir", results_root)

    summary_csv = results_root / "summary.csv"
    if not summary_csv.exists():
        msg = "[suite] Skipping plots (summary.csv not found; run 'eval' stage first)."
        typer.echo(msg)
        notes.append(str(msg))
        status = "skipped"
        reason = "missing_summary_csv"

        mt, mt_utc = _safe_mtime(summary_csv)
        _write_plots_manifest(
            manifest_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": status,
                "reason": reason,
                "runs_root": str(root),
                "runs_root_exists": bool(root.exists()),
                "results_dir": str(results_root),
                "summary_csv_path": _rel(summary_csv, results_root),
                "summary_csv_mtime": mt,
                "summary_csv_mtime_utc": mt_utc,
                "n_written": 0,
                "written": [],
                "notes": notes,
            },
        )
        return

    runs_root_exists = bool(root.exists())
    if not runs_root_exists:
        msg = f"[suite] Runs root missing; timing plots skipped: {root}"
        typer.echo(msg)
        notes.append(str(msg))

    removed = _remove_obsolete_plots(plots_root)
    if removed:
        removed_rel = [_rel(p, results_root) for p in removed]
        msg = "[suite] Removed obsolete plots: " + ", ".join(removed_rel)
        typer.echo(msg)
        notes.append(str(msg))

    try:
        from irl.visualization.paper_figures import (
            load_eval_by_step_table,
            load_eval_summary_table,
            paper_method_groups,
            plot_eval_auc_bars_by_env,
            plot_eval_auc_time_bars_by_env,
            plot_eval_bars_by_env,
            plot_eval_curves_by_env,
            plot_glpe_state_gate_map,
            plot_steps_to_beat_by_env,
        )

        df_summary = load_eval_summary_table(summary_csv)
        methods_present = sorted(set(df_summary["method_key"].tolist()))
        baselines, ablations = paper_method_groups(methods_present)

        all_methods: List[str] = []
        for m in list(baselines) + list(ablations):
            if m not in all_methods:
                all_methods.append(m)

        timing_groups: Dict[str, Dict[str, List[Path]]]
        timing_groups = _groups_for_timing(root) if runs_root_exists else {}

        written.extend(
            plot_eval_bars_by_env(
                df_summary,
                plots_root=plots_root,
                methods_to_plot=baselines,
                title="Task performance (evaluation)",
                filename_suffix="paper_baselines",
            )
            or []
        )
        written.extend(
            plot_eval_bars_by_env(
                df_summary,
                plots_root=plots_root,
                methods_to_plot=ablations,
                title="Ablation study (evaluation)",
                filename_suffix="paper_ablations",
            )
            or []
        )

        raw_path = results_root / "summary_raw.csv"
        raw_for_overlay = raw_path if raw_path.exists() else None

        by_step_path = results_root / "summary_by_step.csv"
        if by_step_path.exists():
            df_steps = load_eval_by_step_table(by_step_path)
            written.extend(
                plot_eval_curves_by_env(
                    df_steps,
                    plots_root=plots_root,
                    methods_to_plot=baselines,
                    title="Learning curves (evaluation)",
                    filename_suffix="paper_baselines_curves",
                    summary_raw_csv=raw_for_overlay,
                )
                or []
            )
            written.extend(
                plot_eval_curves_by_env(
                    df_steps,
                    plots_root=plots_root,
                    methods_to_plot=ablations,
                    title="Ablations over training (evaluation)",
                    filename_suffix="paper_ablations_curves",
                    summary_raw_csv=raw_for_overlay,
                )
                or []
            )

            written.extend(
                plot_eval_auc_bars_by_env(
                    df_steps,
                    plots_root=plots_root,
                    methods_to_plot=all_methods,
                    title="Sample efficiency (AUC of eval return)",
                    filename_suffix="paper_all_methods",
                )
                or []
            )
            written.extend(
                plot_eval_auc_time_bars_by_env(
                    df_steps,
                    timing_groups=timing_groups,
                    plots_root=plots_root,
                    methods_to_plot=all_methods,
                    title="Time efficiency (AUC of eval return)",
                    filename_suffix="paper_all_methods",
                )
                or []
            )

            p = plot_steps_to_beat_by_env(
                df_steps,
                plots_root=plots_root,
                methods_to_plot=baselines,
                title="Steps to beat score threshold (GLPE vs baselines)",
                filename_suffix="paper_baselines",
                summary_raw_csv=raw_path if raw_path.exists() else None,
            )
            if p is not None:
                written.append(Path(p))

            p = plot_steps_to_beat_by_env(
                df_steps,
                plots_root=plots_root,
                methods_to_plot=ablations,
                title="Steps to beat score threshold (GLPE ablations)",
                filename_suffix="paper_ablations",
                summary_raw_csv=raw_path if raw_path.exists() else None,
            )
            if p is not None:
                written.append(Path(p))
        else:
            msg = "[suite] summary_by_step.csv not found; skipping eval learning curves."
            typer.echo(msg)
            notes.append(str(msg))

        traj_root = results_root / "plots" / "trajectories"
        if traj_root.exists():
            written.extend(plot_glpe_state_gate_map(traj_root=traj_root, plots_root=plots_root) or [])
        else:
            msg = "[suite] No trajectories found; skipping GLPE trajectory plots."
            typer.echo(msg)
            notes.append(str(msg))

        try:
            from irl.visualization.paper.training_plots import plot_training_reward_decomposition

            out_paths = plot_training_reward_decomposition(timing_groups, plots_root=plots_root)
            written.extend(out_paths or [])
            if not out_paths:
                msg = "[suite] Training reward plots skipped (no data)."
                typer.echo(msg)
                notes.append(str(msg))
        except Exception as exc:
            msg = f"[suite] Training reward plots skipped ({type(exc).__name__}: {exc})"
            typer.echo(msg)
            notes.append(str(msg))

        try:
            from irl.visualization.timing_figures import plot_timing_breakdown

            out_paths = plot_timing_breakdown(timing_groups, plots_root=plots_root, tail_frac=0.25)
            written.extend(out_paths or [])
        except Exception as exc:
            msg = f"[suite] Timing plots skipped ({type(exc).__name__}: {exc})"
            typer.echo(msg)
            notes.append(str(msg))

        try:
            from irl.visualization.timing_figures import plot_intrinsic_taper_weight

            taper_paths = plot_intrinsic_taper_weight(timing_groups, plots_root=plots_root)
            written.extend(taper_paths or [])
            if not taper_paths:
                msg = "[suite] Intrinsic taper plot skipped (no taper active)."
                typer.echo(msg)
                notes.append(str(msg))
        except Exception as exc:
            msg = f"[suite] Intrinsic taper plot skipped ({type(exc).__name__}: {exc})"
            typer.echo(msg)
            notes.append(str(msg))

    except Exception as exc:
        msg = f"[suite] Plotting failed ({type(exc).__name__}: {exc})"
        typer.echo(msg)
        notes.append(str(msg))
        status = "error"
        reason = f"{type(exc).__name__}: {exc}"

    written = _dedup_paths(written)

    if status == "ok" and not written:
        status = "empty"
        reason = "no_outputs"
        msg = "[suite] WARN  Plot stage produced no outputs (see plots_manifest.json)."
        typer.echo(msg)
        notes.append(str(msg))

    if written:
        typer.echo(f"[suite] Plots written under: {_rel(plots_root, results_root)}")

    mt, mt_utc = _safe_mtime(summary_csv)
    out_list = [_rel(p, results_root) for p in written]

    _write_plots_manifest(
        manifest_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": status,
            "reason": reason,
            "runs_root": str(root),
            "runs_root_exists": bool(runs_root_exists),
            "results_dir": str(results_root),
            "summary_csv_path": _rel(summary_csv, results_root),
            "summary_csv_mtime": mt,
            "summary_csv_mtime_utc": mt_utc,
            "n_written": int(len(out_list)),
            "written": out_list,
            "notes": notes,
        },
    )
