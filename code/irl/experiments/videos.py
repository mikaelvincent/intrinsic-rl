from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import typer

from irl.video import render_rollout_video

_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


def _discover_run_dirs_with_checkpoints(runs_root: Path) -> list[Path]:
    root = Path(runs_root).resolve()
    if not root.exists():
        return []

    seen: set[Path] = set()
    out: list[Path] = []

    for ckpt_dir in root.rglob("checkpoints"):
        if not ckpt_dir.is_dir():
            continue
        try:
            has_any = any(
                p.is_file() and _CKPT_RE.match(p.name) for p in ckpt_dir.iterdir()
            )
        except Exception:
            has_any = False

        if not has_any:
            continue

        run_dir = ckpt_dir.parent.resolve()
        if run_dir not in seen:
            seen.add(run_dir)
            out.append(run_dir)

    return sorted(out, key=lambda p: str(p))


def _list_step_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return []

    out: list[tuple[int, Path]] = []
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
        out.append((step, p))

    out.sort(key=lambda t: (t[0], str(t[1])))
    return out


def _normalize_seeds(seeds: Iterable[int] | None) -> list[int]:
    if seeds is None:
        return [100]
    out: list[int] = []
    for s in seeds:
        try:
            out.append(int(s))
        except Exception:
            continue
    return out or [100]


def run_video_suite(
    runs_root: Path,
    results_dir: Path,
    device: str,
    *,
    policy_mode: str = "mode",
    eval_seeds: Sequence[int] | None = None,
    max_steps: int = 1000,
    fps: int = 30,
    overwrite: bool = False,
) -> None:
    pm = str(policy_mode).strip().lower()
    if pm not in {"mode", "sample"}:
        raise typer.BadParameter("--policy must be one of: mode, sample")

    root = Path(runs_root).resolve()
    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    run_dirs = _discover_run_dirs_with_checkpoints(root)
    if not run_dirs:
        typer.echo(f"[suite] No run directories with checkpoints under {root}")
        return

    seeds = _normalize_seeds(eval_seeds)

    videos_root = (Path(results_dir) / "videos").resolve()
    videos_root.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"[suite] Rendering rollout videos for {len(run_dirs)} run(s) "
        f"({pm}, eval_seeds={seeds}) to {videos_root}"
    )

    for rd in run_dirs:
        ckpts = _list_step_checkpoints(rd)
        if not ckpts:
            typer.echo(f"[suite]    - {rd.name}: no step checkpoints found, skipping")
            continue

        out_dir = videos_root / rd.name
        out_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"[suite]    - {rd.name}: {len(ckpts)} checkpoint(s)")

        for step, ckpt_path in ckpts:
            step_tag = f"step{int(step):09d}"
            for s in seeds:
                out_name = f"{step_tag}__{pm}__evalseed{int(s)}.mp4"
                out_path = out_dir / out_name

                if out_path.exists() and not overwrite:
                    continue

                typer.echo(f"[suite]        * {ckpt_path.name} -> {out_path.name}")
                try:
                    render_rollout_video(
                        ckpt_path=ckpt_path,
                        out_path=out_path,
                        seed=int(s),
                        max_steps=int(max_steps),
                        device=str(device),
                        policy_mode=pm,
                        fps=int(fps),
                    )
                except Exception as exc:
                    typer.echo(f"[warn] Failed to render {ckpt_path}: {exc}")
