"""Training suite helpers for intrinsic-rl experiments.

This module exposes :func:`run_training_suite`, which trains all eligible
configuration files under a configs/ tree.
"""

from __future__ import annotations

import glob
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence

import typer

from irl.cfg import load_config
from irl.cfg.schema import Config
from irl.trainer import train as run_train
from irl.utils.checkpoint import load_checkpoint


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
