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
from typing import List, Optional, Sequence

# Ensure a non-interactive backend for headless environments before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402

from irl.cfg import load_config
from irl.cfg.schema import Config
from irl.plot import _aggregate_runs, _parse_run_name
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

    for rd in run_dirs:
        ckpt = _find_latest_ckpt(rd)
        if ckpt is None:
            typer.echo(f"[suite]  - {rd.name}: no checkpoints found, skipping")
            continue
        typer.echo(f"[suite]  - {rd.name}: ckpt={ckpt.name}, episodes={episodes}")
        try:
            res = _evaluate_ckpt(ckpt, episodes=episodes, device=device)
            results.append(res)
        except Exception as exc:
            typer.echo(f"[suite]    ! evaluation failed: {exc}")

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


def run_plots_suite(
    runs_root: Path,
    results_dir: Path,
    metric: str,
    smooth: int,
    shade: bool,
) -> None:
    """Generate per-environment overlay plots from suite runs.

    Parameters
    ----------
    runs_root : Path
        Root directory that holds individual run subdirectories.
    results_dir : Path
        Directory where the ``plots/`` subdirectory is created.
    metric : str
        Scalar metric name from ``scalars.csv`` to plot (for example
        ``"reward_total_mean"``).
    smooth : int
        Moving-average window (in logged points) applied to each run
        before aggregation. A value of ``1`` disables smoothing.
    shade : bool
        If ``True``, shade a ±1 standard deviation band around each
        mean curve when at least two runs are available for a method.

    Returns
    -------
    None
        Plot images are written under ``results_dir / "plots"``.
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
    groups: dict[str, dict[str, List[Path]]] = {}
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

    for env_id, by_method in sorted(groups.items(), key=lambda kv: kv[0]):
        if not by_method:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        any_plotted = False

        for method, dirs in sorted(by_method.items(), key=lambda kv: kv[0]):
            try:
                agg = _aggregate_runs(dirs, metric=metric, smooth=int(smooth))
            except Exception as exc:
                typer.echo(
                    f"[suite] Plot skip for env={env_id}, method={method}: aggregate error ({exc})"
                )
                continue

            label = f"{method} (n={agg.n_runs})"
            ax.plot(agg.steps, agg.mean, label=label)
            any_plotted = True

            if shade and agg.n_runs >= 2 and agg.std.size > 0:
                lo = agg.mean - agg.std
                hi = agg.mean + agg.std
                ax.fill_between(agg.steps, lo, hi, alpha=0.2, linewidth=0)

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Environment steps")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{env_id} — {metric}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        env_tag = env_id.replace("/", "-")
        out = plots_root / f"{env_tag}__overlay_{metric}.png"
        tmp = out.with_suffix(out.suffix + ".tmp")
        fmt = out.suffix.lstrip(".") or "png"
        fig.savefig(str(tmp), dpi=150, bbox_inches="tight", format=fmt)
        atomic_replace(tmp, out)
        plt.close(fig)
        typer.echo(f"[suite] Saved overlay plot: {out}")


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
        "-n",
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
        "-s",
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
        "-n",
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
    metric: str = typer.Option(
        "reward_total_mean",
        "--metric",
        "-m",
        help="Scalar metric name from scalars.csv to plot.",
    ),
    smooth: int = typer.Option(
        5,
        "--smooth",
        "-s",
        help="Moving-average window (in logged points) for smoothing.",
    ),
    shade: bool = typer.Option(
        True,
        "--shade/--no-shade",
        help="Shade ±1 std as a translucent band when ≥2 runs are available.",
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
        "-n",
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
        "-s",
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
        "-n",
        help="Number of evaluation episodes per checkpoint.",
    ),
    results_dir: Path = typer.Option(
        Path("results_suite"),
        "--results-dir",
        "-o",
        help="Directory to write summaries and plots.",
    ),
    metric: str = typer.Option(
        "reward_total_mean",
        "--metric",
        "-m",
        help="Scalar metric name from scalars.csv to plot.",
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
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="When enabled, resume from existing checkpoints and skip runs that already reached total_steps.",
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
    )

    eval_device = device if device is not None else "cpu"
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
