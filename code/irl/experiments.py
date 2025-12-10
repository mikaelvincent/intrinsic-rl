"""Experiment suite runner for intrinsic-rl.

This module provides a small Typer-based CLI that can:

  * Train all eligible configuration files under a configs/ tree.
  * Evaluate the latest checkpoint for each run directory.
  * Generate simple per-environment overlay plots.

Typical usage from the repo's `code/` directory:

    # One-shot: train all configs, then eval + plots
    python -m irl.experiments full

Or individual stages:

    python -m irl.experiments train
    python -m irl.experiments eval
    python -m irl.experiments plots
"""

from __future__ import annotations

import glob
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Dict

# Ensure a non-interactive backend for headless environments before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402
import torch  # noqa: E402

from irl.cfg import load_config
from irl.cfg.schema import Config
from irl.plot import _aggregate_runs, _parse_run_name, plot_normalized_summary, plot_trajectory_heatmap
from irl.sweep import (
    RunResult,
    _aggregate,
    _evaluate_ckpt,
    _find_latest_ckpt,
    _write_raw_csv,
    _write_summary_csv,
)
from irl.trainer import train as run_train
from irl.utils.checkpoint import atomic_replace, load_checkpoint
from irl.evaluator import evaluate

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


def _discover_configs(
    configs_root: Path,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> List[Path]:
    """Return sorted list of config paths under configs_root.

    Parameters
    ----------
    configs_root:
        Root directory that holds YAML configs (e.g. code/configs).
    include:
        Optional glob patterns **relative** to configs_root. If omitted,
        defaults to ["**/*.yaml", "**/*.yml"].
    exclude:
        Optional glob patterns to subtract from the result.
    """
    root = configs_root.resolve()
    if not root.exists():
        raise typer.BadParameter(f"configs_dir does not exist: {root}")

    patterns = list(include) if include else ["**/*.yaml", "**/*.yml"]
    candidates: set[Path] = set()

    for pat in patterns:
        full_pattern = str(root / pat)
        for hit in glob.glob(full_pattern, recursive=True):
            p = Path(hit)
            if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}:
                candidates.add(p.resolve())

    if exclude:
        to_drop: set[Path] = set()
        for pat in exclude:
            full_pattern = str(root / pat)
            for hit in glob.glob(full_pattern, recursive=True):
                to_drop.add(Path(hit).resolve())
        candidates.difference_update(to_drop)

    return sorted(candidates)


def _run_dir_for(cfg: Config, cfg_path: Path, seed: int, runs_root: Path) -> Path:
    """Deterministic run directory for (config, seed).

    Layout (relative to runs_root):

    <method>__<env_id>__seed<seed>__<config_stem>

    where env_id has '/' replaced by '-'.
    """
    env_tag = str(cfg.env.id).replace("/", "-")
    method = str(cfg.method)
    stem = cfg_path.stem
    name = f"{method}__{env_tag}__seed{int(seed)}__{stem}"
    return runs_root / name


def _format_steps(step: int) -> str:
    """Human-friendly representation of a step count."""
    if step >= 1_000_000:
        return f"{step / 1_000_000:.1f}M"
    if step >= 1_000:
        return f"{step / 1_000:.1f}k"
    return str(step)


def run_training_suite(
    configs_dir: Path,
    include: Sequence[str],
    exclude: Sequence[str],
    total_steps: int,
    runs_root: Path,
    seeds: Sequence[int],
    device: Optional[str],
    resume: bool,
    auto_async: bool = False,
) -> None:
    """Train all (config, seed) combinations into a suite of runs.

    Parameters
    ----------
    configs_dir : Path
        Root directory containing YAML configuration files.
    include : Sequence[str]
        Glob patterns (relative to ``configs_dir``) selecting which configs
        to run. When empty, all ``*.yaml``/``*.yml`` files are considered.
    exclude : Sequence[str]
        Glob patterns (relative to ``configs_dir``) that are excluded from
        the included set.
    total_steps : int
        Default target environment steps per run. If a configuration
        provides ``exp.total_steps``, that value takes precedence for
        that particular run.
    runs_root : Path
        Root directory into which per-run subdirectories are created.
    seeds : Sequence[int]
        Optional list of seeds to use for each config. When empty, the
        seed stored in the configuration is used instead.
    device : str or None
        Optional device override for training (for example ``"cpu"`` or
        ``"cuda:0"``). When ``None``, each config's ``device`` field is
        respected.
    resume : bool
        When ``True``, resume from existing checkpoints (if present) and
        skip runs that already reached their target step budget.
    auto_async : bool
        When ``True``, automatically enable ``async_vector`` for configs that
        request more than one vector environment and have not explicitly set
        ``env.async_vector``. When ``False``, the value from each config is
        respected as-is.

    Returns
    -------
    None
        The function is called for its side effects: training runs are
        executed and checkpoints/logs are written under ``runs_root``.
    """
    cfg_paths = _discover_configs(configs_dir, include=include, exclude=exclude)
    if not cfg_paths:
        typer.echo(f"[suite] No configuration files found under {configs_dir}")
        return

    typer.echo(f"[suite] Found {len(cfg_paths)} config(s) under {configs_dir}")
    runs_root.mkdir(parents=True, exist_ok=True)

    for cfg_path in cfg_paths:
        try:
            cfg = load_config(str(cfg_path))
        except Exception as exc:
            typer.echo(f"[suite] Skipping {cfg_path}: failed to load config ({exc})")
            continue

        seed_list = list(seeds) if seeds else [int(cfg.seed)]
        for seed_val in seed_list:
            cfg_seeded = replace(cfg, seed=int(seed_val))
            if device is not None:
                cfg_seeded = replace(cfg_seeded, device=str(device))

            # Prefer per-config exp.total_steps when provided; otherwise, use CLI/default.
            steps_from_cfg = getattr(getattr(cfg_seeded, "exp", object()), "total_steps", None)
            target_steps = int(steps_from_cfg) if steps_from_cfg is not None else int(total_steps)

            # Optionally enable AsyncVectorEnv automatically when multiple envs are requested.
            if auto_async and int(cfg_seeded.env.vec_envs) > 1 and not bool(
                getattr(cfg_seeded.env, "async_vector", False)
            ):
                cfg_seeded = replace(cfg_seeded, env=replace(cfg_seeded.env, async_vector=True))
                typer.echo(
                    f"[suite]    -> enabling AsyncVectorEnv (num_envs={cfg_seeded.env.vec_envs}) for {cfg_path.name}"
                )

            run_dir = _run_dir_for(cfg_seeded, cfg_path, seed_val, runs_root)

            latest_ckpt = run_dir / "checkpoints" / "ckpt_latest.pt"
            existing_step = 0
            if latest_ckpt.exists() and resume:
                try:
                    payload = load_checkpoint(latest_ckpt, map_location="cpu")
                    existing_step = int(payload.get("step", 0))
                except Exception:
                    existing_step = 0

            if resume and existing_step >= target_steps:
                typer.echo(
                    f"[suite] SKIP  {cfg_path.name} "
                    f"(method={cfg_seeded.method}, env={cfg_seeded.env.id}, seed={seed_val}) "
                    f"— already at step {existing_step} ≥ {target_steps}"
                )
                continue

            resume_flag = resume and latest_ckpt.exists() and existing_step > 0
            mode = "resume" if resume_flag else "fresh"
            typer.echo(
                f"[suite] TRAIN {cfg_path.name} "
                f"(method={cfg_seeded.method}, env={cfg_seeded.env.id}, seed={seed_val}) "
                f"[{mode}, from step {_format_steps(existing_step)} → {_format_steps(target_steps)}, "
                f"device={cfg_seeded.device}]"
            )
            run_train(
                cfg_seeded,
                total_steps=int(target_steps),
                run_dir=run_dir,
                resume=resume_flag,
            )


def run_eval_suite(
    runs_root: Path,
    results_dir: Path,
    episodes: int,
    device: str,
) -> None:
    """Evaluate latest checkpoints for all run directories.

    Parameters
    ----------
    runs_root : Path
        Root directory that holds individual run subdirectories.
    results_dir : Path
        Directory where evaluation CSV files will be written.
    episodes : int
        Number of evaluation episodes to run per checkpoint.
    device : str
        Device string to use for evaluation (for example ``"cpu"`` or
        ``"cuda:0"``).

    Returns
    -------
    None
        Evaluation summaries are written to ``results_dir`` as
        ``summary_raw.csv`` and ``summary.csv``.
    """
    root = runs_root.resolve()
    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not run_dirs:
        typer.echo(f"[suite] No run directories under {root}")
        return

    typer.echo(f"[suite] Evaluating {len(run_dirs)} run(s) from {root}")
    results: List[RunResult] = []

    # Also save trajectory files for the plots/ directory
    traj_dir = results_dir / "plots" / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    for rd in run_dirs:
        ckpt = _find_latest_ckpt(rd)
        if ckpt is None:
            typer.echo(f"[suite]    - {rd.name}: no checkpoints found, skipping")
            continue
        typer.echo(f"[suite]    - {rd.name}: ckpt={ckpt.name}, episodes={episodes}")
        try:
            # We call evaluate directly to capture stats, AND set save_traj=True to populate files for heatmap
            summary = evaluate(
                env=str(_parse_run_name(rd).get("env", "UnknownEnv")), # best effort env, evaluate uses checkpoint anyway
                ckpt=ckpt, 
                episodes=episodes, 
                device=device,
                save_traj=True,
                traj_out_dir=traj_dir
            )
            
            # Construct RunResult manually since we bypassed _evaluate_ckpt
            payload = load_checkpoint(ckpt, map_location="cpu")
            step = int(payload.get("step", -1))
            
            # Re-parse method/env/seed from the run directory name for the CSV
            info = _parse_run_name(rd)
            
            res = RunResult(
                method=info.get("method", "unknown"),
                env_id=summary["env_id"],
                seed=int(info.get("seed", 0)),
                ckpt_path=ckpt,
                ckpt_step=step,
                episodes=int(summary["episodes"]),
                mean_return=float(summary["mean_return"]),
                std_return=float(summary["std_return"]),
                min_return=float(summary["min_return"]),
                max_return=float(summary["max_return"]),
                mean_length=float(summary["mean_length"]),
                std_length=float(summary["std_length"]),
            )
            results.append(res)
        except Exception as exc:
            typer.echo(f"[suite]        ! evaluation failed: {exc}")

    if not results:
        typer.echo("[suite] No checkpoints evaluated; nothing to write.")
        return

    results_root = results_dir.resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    raw_path = results_root / "summary_raw.csv"
    summary_path = results_root / "summary.csv"

    _write_raw_csv(results, raw_path)
    agg_rows = _aggregate(results)
    _write_summary_csv(agg_rows, summary_path)

    typer.echo(f"[suite] Wrote per-run results to {raw_path}")
    typer.echo(f"[suite] Wrote aggregated summary to {summary_path}")


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
            # - Proposed method gets higher z-order, thicker line, and full alpha.
            # - Baselines are slightly more transparent and thinner to reduce clutter.
            is_main_proposed = method.lower() == "proposed"
            
            lw = 2.5 if is_main_proposed else 1.5
            alpha = 1.0 if is_main_proposed else 0.75
            zorder = 10 if is_main_proposed else 2
            
            label = f"{method} (n={agg.n_runs})"
            ax.plot(agg.steps, agg.mean, label=label, linewidth=lw, alpha=alpha, zorder=zorder)
            any_plotted = True

            if shade and agg.n_runs >= 2 and agg.std.size > 0:
                lo = agg.mean - agg.std
                hi = agg.mean + agg.std
                ax.fill_between(agg.steps, lo, hi, alpha=0.15, linewidth=0, zorder=zorder-1)

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
        ax1.set_ylabel("Extrinsic Reward", color=color1, fontweight='bold')
        ax1.plot(agg_rew.steps, agg_rew.mean, color=color1, linewidth=2.0, label="Reward")
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Shade reward variance
        if agg_rew.n_runs >= 2 and agg_rew.std.size > 0:
            ax1.fill_between(
                agg_rew.steps, 
                agg_rew.mean - agg_rew.std, 
                agg_rew.mean + agg_rew.std, 
                color=color1, alpha=0.15, linewidth=0
            )

        # Right Axis: Gate Rate (Red)
        ax2 = ax1.twinx()
        color2 = "tab:red"
        # If mean is small (<1.1), assume fraction 0-1, label as Rate. Else %, label as %.
        is_pct = (agg_gate.mean.max() > 1.1)
        ylabel = "Gate Rate (%)" if is_pct else "Gate Rate (0-1)"
        if not is_pct:
            # Force 0-1 limits if fraction, for clarity
            ax2.set_ylim(0, 1.05)
        
        ax2.set_ylabel(ylabel, color=color2, fontweight='bold')
        ax2.plot(agg_gate.steps, agg_gate.mean, color=color2, linewidth=2.0, linestyle="--", label="Gate Rate")
        ax2.tick_params(axis='y', labelcolor=color2)

        # Shade gate variance
        if agg_gate.n_runs >= 2 and agg_gate.std.size > 0:
            ax2.fill_between(
                agg_gate.steps, 
                agg_gate.mean - agg_gate.std, 
                agg_gate.mean + agg_gate.std, 
                color=color2, alpha=0.15, linewidth=0
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
            alpha=0.9
        )
        if agg_imp.n_runs >= 2 and agg_imp.std.size > 0:
            ax.fill_between(
                agg_imp.steps, 
                agg_imp.mean - agg_imp.std, 
                agg_imp.mean + agg_imp.std, 
                color="#1f77b4", 
                alpha=0.15, 
                linewidth=0
            )

        # Plot LP RMS
        ax.plot(
            agg_lp.steps, 
            agg_lp.mean, 
            label="LP (Competence)", 
            color="#ff7f0e",  # Safety Orange
            linewidth=2.5,
            alpha=0.9
        )
        if agg_lp.n_runs >= 2 and agg_lp.std.size > 0:
            ax.fill_between(
                agg_lp.steps, 
                agg_lp.mean - agg_lp.std, 
                agg_lp.mean + agg_lp.std, 
                color="#ff7f0e", 
                alpha=0.15, 
                linewidth=0
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
      3. Ablation Study: Proposed vs Variants (reward_mean).
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
    
    # Similarly for ablations, the full method should be the reference point on top.
    ablations = [
        "proposed_global_rms",
        "proposed_lp_only",
        "proposed_impact_only",
        "proposed_nogate",
        "proposed",
    ]

    # 1. Main Comparison (Extrinsic)
    # Task performance. Smoothed reasonably to show trends.
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
    # The optimization objective. Since intrinsic rewards can be noisy/spiky,
    # we use heavier smoothing here to visualize the "dense gradient" landscape clearly.
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

    # 3. Ablation Study
    # Component analysis using extrinsic reward.
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

    # 4. Gating Dynamics Plot (Dual Axis)
    # Shows mechanism: correlation between Extrinsic Reward and Gate Rate.
    # Uses higher smoothing (25) to highlight the macroscopic phase shifts.
    _generate_gating_plot(
        groups,
        plots_root=plots_root,
        smooth=25,
    )

    # 5. Intrinsic Component Evolution (New)
    # Shows the balance between Novelty and Competence over time.
    _generate_component_plot(
        groups,
        plots_root=plots_root,
        smooth=25,
    )

    # 6. Normalized Performance Summary (Bar Chart)
    # Requires summary.csv to exist (generated by `eval-many`)
    summary_csv = results_dir / "summary.csv"
    if summary_csv.exists():
        bar_plot_path = plots_root / "summary_normalized_bars.png"
        plot_normalized_summary(summary_csv, bar_plot_path, highlight_method="proposed")
        typer.echo(f"[suite] Saved normalized summary bars: {bar_plot_path}")
    else:
        typer.echo("[suite] Skipping bar chart (summary.csv not found; run 'eval' stage first).")

    # 7. Trajectory Heatmaps
    _generate_trajectory_plots(results_dir, plots_root)


@app.command("train")
def cli_train(
    configs_dir: Path = typer.Option(
        Path("configs"),
        "--configs-dir",
        "-c",
        help="Root directory containing YAML configs (scanned recursively).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    include: List[str] = typer.Option(
        [],
        "--include",
        "-i",
        help="Glob(s) relative to configs_dir to select configs (e.g. 'mountaincar_*.yaml'). "
        "If omitted, all *.yaml/ *.yml files are used.",
    ),
    exclude: List[str] = typer.Option(
        [],
        "--exclude",
        "-x",
        help="Glob(s) relative to configs_dir to exclude (e.g. 'mujoco/*_debug.yaml').",
    ),
    total_steps: int = typer.Option(
        150_000,
        "--total-steps",
        "-t",
        help="Default target environment steps per run (overridden by exp.total_steps in config if present).",
    ),
    runs_root: Path = typer.Option(
        Path("runs_suite"),
        "--runs-root",
        help="Root directory where suite run subdirectories will be created.",
    ),
    seed: List[int] = typer.Option(
        [],
        "--seed",
        "-r",
        help="Override seeds to run for every config (repeatable). "
        "If omitted, each config's own seed is used.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help='Override device for training (e.g. "cpu" or "cuda:0"). '
        "Defaults to each config's device field.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="When enabled, resume from existing checkpoints and skip runs that already reached total_steps.",
    ),
    auto_async: bool = typer.Option(
        False,
        "--auto-async/--no-auto-async",
        help="When enabled, auto-enable async vector envs for configs requesting multiple environments.",
    ),
) -> None:
    """Train all eligible configs into a deterministic runs_root."""
    run_training_suite(
        configs_dir=configs_dir,
        include=include,
        exclude=exclude,
        total_steps=total_steps,
        runs_root=runs_root,
        seeds=seed,
        device=device,
        resume=resume,
        auto_async=auto_async,
    )


@app.command("eval")
def cli_eval(
    runs_root: Path = typer.Option(
        Path("runs_suite"),
        "--runs-root",
        help="Root directory that holds suite run subdirectories.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    results_dir: Path = typer.Option(
        Path("results_suite"),
        "--results-dir",
        "-o",
        help="Directory to write summary_raw.csv and summary.csv.",
    ),
    episodes: int = typer.Option(
        5,
        "--episodes",
        "-e",
        help="Number of evaluation episodes per checkpoint.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help='Device to use for evaluation (e.g. "cpu" or "cuda:0").',
    ),
) -> None:
    """Evaluate latest checkpoints for all suite runs."""
    run_eval_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        episodes=episodes,
        device=device,
    )


@app.command("plots")
def cli_plots(
    runs_root: Path = typer.Option(
        Path("runs_suite"),
        "--runs-root",
        help="Root directory that holds suite run subdirectories.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    results_dir: Path = typer.Option(
        Path("results_suite"),
        "--results-dir",
        "-o",
        help="Directory where plots/ will be created.",
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        "-m",
        help="Scalar metric to plot. If omitted, generates standard paper plots (Extrinsic, Total, Ablation).",
    ),
    smooth: int = typer.Option(
        5,
        "--smooth",
        "-s",
        help="Moving-average window (in logged points).",
    ),
    shade: bool = typer.Option(
        True,
        "--shade/--no-shade",
        help="Shade ±1 std band on overlay plots when ≥2 runs are available.",
    ),
) -> None:
    """Generate per-environment overlay plots from suite runs."""
    run_plots_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        metric=metric,
        smooth=smooth,
        shade=shade,
    )


@app.command("full")
def cli_full(
    configs_dir: Path = typer.Option(
        Path("configs"),
        "--configs-dir",
        "-c",
        help="Root directory containing YAML configs (scanned recursively).",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    include: List[str] = typer.Option(
        [],
        "--include",
        "-i",
        help="Glob(s) relative to configs_dir to select configs (e.g. 'mountaincar_*.yaml'). "
        "If omitted, all *.yaml/ *.yml files are used.",
    ),
    exclude: List[str] = typer.Option(
        [],
        "--exclude",
        "-x",
        help="Glob(s) relative to configs_dir to exclude.",
    ),
    total_steps: int = typer.Option(
        150_000,
        "--total-steps",
        "-t",
        help="Default step budget per run (overridden by exp.total_steps in config if present).",
    ),
    runs_root: Path = typer.Option(
        Path("runs_suite"),
        "--runs-root",
        help="Root directory where suite run subdirectories will be created.",
    ),
    seed: List[int] = typer.Option(
        [],
        "--seed",
        "-r",
        help="Override seeds to run for every config (repeatable). "
        "If omitted, each config's own seed is used.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        "-d",
        help='Override device for training/evaluation (e.g. "cpu" or "cuda:0"). '
        "Defaults to each config's device field for training, and 'cpu' for eval if unset.",
    ),
    episodes: int = typer.Option(
        5,
        "--episodes",
        "-e",
        help="Number of evaluation episodes per checkpoint.",
    ),
    results_dir: Path = typer.Option(
        Path("results_suite"),
        "--results-dir",
        "-o",
        help="Directory to write summaries and plots.",
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        "-m",
        help="Scalar metric to plot. If omitted, generates standard paper plots.",
    ),
    smooth: int = typer.Option(
        5,
        "--smooth",
        "-w",
        help="Moving-average window (in logged points).",
    ),
    shade: bool = typer.Option(
        True,
        "--shade/--no-shade",
        help="Shade ±1 std band on overlay plots when ≥2 runs are available.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="When enabled, resume from existing checkpoints and skip runs that already reached total_steps.",
    ),
    auto_async: bool = typer.Option(
        False,
        "--auto-async/--no-auto-async",
        help="When enabled, auto-enable async vector envs for configs requesting multiple environments.",
    ),
) -> None:
    """Run training, evaluation, and plotting in one shot."""
    run_training_suite(
        configs_dir=configs_dir,
        include=include,
        exclude=exclude,
        total_steps=total_steps,
        runs_root=runs_root,
        seeds=seed,
        device=device,
        resume=resume,
        auto_async=auto_async,
    )

    if device is None:
        eval_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        eval_device = device

    run_eval_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        episodes=episodes,
        device=eval_device,
    )

    run_plots_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        metric=metric,
        smooth=smooth,
        shade=shade,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
