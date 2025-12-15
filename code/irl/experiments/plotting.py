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

Diagnostics & transparency
--------------------------
This module is intentionally verbose when run as part of the experiment suite:

* It lists directories found under the runs root and results directory.
* It prints a final report listing directories that were skipped or not processed,
  together with a best-effort reason.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer  # noqa: E402

from irl.plot import (
    _parse_run_name,
    plot_normalized_summary,
)
from irl.utils.checkpoint import load_checkpoint
from irl.visualization.data import _read_scalars  # strict CSV validation + de-dupe

from .plot_helpers import (
    _generate_comparison_plot,
    _generate_gating_plot,
    _generate_component_plot,
    _generate_trajectory_plots,
)

_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


def _rel(path: Path, root: Path) -> str:
    """Return path relative to root when possible (for nicer logs)."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _list_dirs(label: str, root: Path) -> List[Path]:
    """List immediate subdirectories under root and echo them."""
    if not root.exists():
        typer.echo(f"[suite] {label}: {root} (missing)")
        return []
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    typer.echo(f"[suite] {label}: {root} ({len(dirs)} directorie(s))")
    for d in dirs:
        typer.echo(f"[suite]   - {d.name}")
    return dirs


def _discover_run_dirs(runs_root: Path) -> List[Path]:
    """Recursively discover run directories that contain logs/scalars.csv."""
    root = runs_root.resolve()
    if not root.exists():
        return []

    run_dirs: list[Path] = []
    seen: set[Path] = set()

    for csv_path in root.rglob("logs/scalars.csv"):
        # Expect <run_dir>/logs/scalars.csv
        try:
            run_dir = csv_path.parent.parent
        except Exception:
            continue
        if not run_dir.is_dir():
            continue
        rd = run_dir.resolve()
        if rd not in seen:
            seen.add(rd)
            run_dirs.append(rd)

    return sorted(run_dirs, key=lambda p: str(p))


def _find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    """Return the best-effort latest checkpoint inside run_dir/checkpoints."""
    ckpt_dir = run_dir / "checkpoints"
    latest = ckpt_dir / "ckpt_latest.pt"
    if latest.exists():
        return latest

    if not ckpt_dir.exists():
        return None

    best_step = -1
    best_path: Optional[Path] = None
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = _CKPT_RE.match(p.name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except Exception:
            continue
        if step > best_step:
            best_step = step
            best_path = p
    return best_path


def _infer_method_env_from_checkpoint(run_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Best-effort (method, env_id, seed) from latest checkpoint cfg."""
    ckpt = _find_latest_ckpt(run_dir)
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

    return (str(method) if method is not None else None), (str(env_id) if env_id is not None else None), seed


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
      6. Normalized Performance Summary (Bar Chart).
      7. Trajectory Heatmaps (State Space).

    Note
    ----
    Standard deviation indicators are disabled. The ``shade`` argument is kept
    for compatibility but ignored by the plotting helpers.

    Additionally, emits a transparent discovery summary listing:
      * all directories under runs_root/results_dir, and
      * which run directories were skipped (and why).
    """
    root = runs_root.resolve()
    results_root = results_dir.resolve()

    # --- Required by the prompt: list directories found ---
    _list_dirs("Runs root", root)
    _list_dirs("Results dir", results_root)

    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    # Discover run dirs recursively (handles nested layouts)
    discovered_run_dirs = _discover_run_dirs(root)
    if not discovered_run_dirs:
        typer.echo(f"[suite] No run directories with logs/scalars.csv under {root}")
        return

    typer.echo(f"[suite] Discovered {len(discovered_run_dirs)} run(s) with logs/scalars.csv")

    # --- Validate and group runs ---
    groups: Dict[str, Dict[str, List[Path]]] = {}
    processed_runs: list[Path] = []
    skipped_runs: dict[Path, str] = {}

    for rd in discovered_run_dirs:
        # 1) Validate scalars.csv is readable and non-empty (this also de-dupes steps).
        try:
            df = _read_scalars(rd)
        except Exception as exc:
            skipped_runs[rd] = f"Unreadable logs/scalars.csv ({type(exc).__name__}: {exc})"
            continue

        if df.empty:
            skipped_runs[rd] = "logs/scalars.csv has no data rows"
            continue

        # 2) Infer env/method from the directory name, then fall back to checkpoint cfg.
        info = _parse_run_name(rd)
        method = info.get("method")
        env = info.get("env")

        if env is None or (method is None or method.strip() == ""):
            m2, e2, _ = _infer_method_env_from_checkpoint(rd)
            method = method or m2
            env = env or e2

        # 3) Canonicalize keys (case-insensitive methods; env '/' -> '-')
        method_key = (str(method) if method is not None else "unknown").strip().lower() or "unknown"
        env_key = (str(env) if env is not None else "unknown_env").strip() or "unknown_env"
        env_key = env_key.replace("/", "-")

        groups.setdefault(env_key, {}).setdefault(method_key, []).append(rd)
        processed_runs.append(rd)

    if not groups:
        typer.echo("[suite] No valid runs to plot (all discovered runs were skipped).")
        # Print skipped report (required)
        typer.echo(f"[suite] Skipped {len(skipped_runs)} run directory(ies):")
        for p, reason in sorted(skipped_runs.items(), key=lambda kv: str(kv[0])):
            typer.echo(f"[suite]   - {_rel(p, root)}: {reason}")
        return

    plots_root = (results_root / "plots").resolve()
    plots_root.mkdir(parents=True, exist_ok=True)

    # Collect all method keys we actually have data for.
    all_methods = sorted({m for m_map in groups.values() for m in m_map})

    if metric is not None:
        # Single-metric mode: plot everything found (all methods discovered)
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
        # --- Paper Mode: Generate specific figures ---

        # Prefer a stable "known baseline" order.
        # IMPORTANT: keep Proposed ablations out of baseline comparison plots;
        # they are plotted in the ablation figures instead.
        preferred = ["vanilla", "icm", "rnd", "ride", "riac"]
        baselines: list[str] = [m for m in preferred if m in all_methods]
        extras = [
            m
            for m in all_methods
            if m not in baselines and m != "proposed" and not m.startswith("proposed_")
        ]
        baselines.extend(extras)
        if "proposed" in all_methods:
            baselines.append("proposed")  # always draw proposed on top (last)

        # Proposed ablations (best-effort): include any method starting with proposed_
        ablation_priority = ["proposed_lp_only", "proposed_impact_only", "proposed_nogate"]
        ablations: list[str] = [m for m in ablation_priority if m in all_methods]
        other_abls = sorted(
            [m for m in all_methods if m.startswith("proposed_") and m not in ablations]
        )
        ablations.extend(other_abls)
        if "proposed" in all_methods:
            ablations.append("proposed")

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

        # 4. Gating Dynamics Plot (Dual Axis) — Proposed only
        _generate_gating_plot(
            groups,
            plots_root=plots_root,
            smooth=25,
        )

        # 5. Intrinsic Component Evolution — Proposed only
        _generate_component_plot(
            groups,
            plots_root=plots_root,
            smooth=25,
        )

        # 6. Normalized Performance Summary (Bar Chart)
        summary_csv = results_root / "summary.csv"
        if summary_csv.exists():
            bar_plot_path = plots_root / "summary_normalized_bars.png"
            plot_normalized_summary(summary_csv, bar_plot_path, highlight_method="proposed")
            typer.echo(f"[suite] Saved normalized summary bars: {bar_plot_path}")
        else:
            typer.echo("[suite] Skipping bar chart (summary.csv not found; run 'eval' stage first).")

        # 7. Trajectory Heatmaps
        _generate_trajectory_plots(results_root, plots_root)

    # --- Final reporting (required) ---
    typer.echo(f"[suite] Plotting inputs: processed {len(processed_runs)} run(s).")

    # Identify top-level dirs that were not processed as runs (and why).
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
