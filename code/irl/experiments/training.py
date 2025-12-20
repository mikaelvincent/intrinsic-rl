from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer

from irl.cfg import load_config
from irl.cfg.schema import Config
from irl.trainer import train as run_train
from irl.utils.checkpoint import load_checkpoint
from irl.utils.steps import resolve_total_steps as _resolve_total_steps


def _discover_configs(configs_root: Path) -> list[Path]:
    root = Path(configs_root).resolve()
    if not root.exists():
        raise typer.BadParameter(f"configs_dir does not exist: {root}")
    if not root.is_dir():
        raise typer.BadParameter(f"configs_dir is not a directory: {root}")

    cfgs = [p for p in root.rglob("*.yaml") if p.is_file()] + [
        p for p in root.rglob("*.yml") if p.is_file()
    ]
    cfgs = [p.resolve() for p in cfgs]
    cfgs.sort(key=lambda p: str(p.relative_to(root)))
    return cfgs


def _run_dir_name(cfg: Config, cfg_path: Path, seed: int) -> str:
    method_tag = str(cfg.method).strip().lower()
    env_tag = str(cfg.env.id).strip().replace("/", "-")
    cfg_tag = str(cfg_path.stem).strip()
    return f"{method_tag}__{env_tag}__seed{int(seed)}__{cfg_tag}"


def _run_dir_for(cfg: Config, cfg_path: Path, seed: int, runs_root: Path) -> Path:
    return Path(runs_root) / _run_dir_name(cfg, cfg_path, seed)


def _format_steps(step: int) -> str:
    if step >= 1_000_000:
        return f"{step / 1_000_000:.1f}M"
    if step >= 1_000:
        return f"{step / 1_000:.1f}k"
    return str(step)


def _normalize_seed_list(seed_like: object) -> list[int]:
    if seed_like is None:
        return []
    if isinstance(seed_like, (list, tuple)):
        raw = [int(s) for s in seed_like]
    else:
        raw = [int(seed_like)]

    out: list[int] = []
    seen: set[int] = set()
    for s in raw:
        if s in seen:
            continue
        out.append(int(s))
        seen.add(int(s))
    return out


def run_training_suite(
    configs_dir: Path,
    runs_root: Path,
) -> None:
    typer.echo(
        f"[suite] Train defaults: configs_dir={Path(configs_dir).resolve()}, "
        f"runs_root={Path(runs_root).resolve()}, total_steps=cfg.exp.total_steps"
    )
    cfg_paths = _discover_configs(configs_dir)
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

        seed_list = _normalize_seed_list(getattr(cfg, "seed", 1))
        if not seed_list:
            typer.echo(f"[suite] Skipping {cfg_path}: seed list is empty")
            continue

        for seed_val in seed_list:
            cfg_seeded = replace(cfg, seed=int(seed_val))

            target_steps = _resolve_total_steps(
                cfg_seeded,
                None,
                default_total_steps=10_000,
                prefer_cfg=True,
                align_to_vec_envs=True,
            )

            run_dir = _run_dir_for(cfg_seeded, cfg_path, seed_val, runs_root)

            latest_ckpt = run_dir / "checkpoints" / "ckpt_latest.pt"
            existing_step = 0
            if latest_ckpt.exists():
                try:
                    payload = load_checkpoint(latest_ckpt, map_location="cpu")
                    existing_step = int(payload.get("step", 0))
                except Exception:
                    existing_step = 0

            if existing_step >= target_steps:
                typer.echo(
                    f"[suite] SKIP  {cfg_path.name} "
                    f"(method={cfg_seeded.method}, env={cfg_seeded.env.id}, seed={seed_val}) "
                    f"— already at step {existing_step} ≥ {target_steps}"
                )
                continue

            resume_flag = bool(latest_ckpt.exists())
            mode = "resume" if resume_flag else "fresh"
            typer.echo(
                f"[suite] TRAIN {cfg_path.name} "
                f"(method={cfg_seeded.method}, env={cfg_seeded.env.id}, seed={seed_val}) "
                f"[{mode}, from step {_format_steps(existing_step)} → {_format_steps(target_steps)}, "
                f"device={cfg_seeded.device}]"
            )
            run_train(
                cfg_seeded,
                total_steps=None,
                run_dir=run_dir,
                resume=resume_flag,
            )
