from __future__ import annotations

from pathlib import Path
from typing import Iterable

import typer

from irl.cli.validators import normalize_policy_mode
from irl.paper_defaults import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_MAX_STEPS,
    DEFAULT_VIDEO_POLICY_MODE,
    DEFAULT_VIDEO_SEEDS,
    RESULTS_DIR,
    RUNS_ROOT,
)
from irl.pipelines.discovery import discover_run_dirs_with_latest_ckpt, discover_run_dirs_with_step_ckpts
from irl.utils.runs import list_step_ckpts
from irl.video import render_rollout_video


def _normalize_seeds(seeds: Iterable[int]) -> list[int]:
    out: list[int] = []
    for s in seeds:
        try:
            out.append(int(s))
        except Exception:
            continue
    return out or [100]


def _discover_run_dirs(root: Path) -> list[Path]:
    step_runs = discover_run_dirs_with_step_ckpts(root)
    latest_runs = [rd for rd, _ in discover_run_dirs_with_latest_ckpt(root)]

    seen: set[Path] = set()
    out: list[Path] = []
    for rd in list(step_runs) + list(latest_runs):
        p = Path(rd).resolve()
        if p not in seen:
            out.append(p)
            seen.add(p)

    return sorted(out, key=lambda p: str(p))


def _ckpts_for_run(run_dir: Path) -> list[tuple[str, Path]]:
    step_ckpts = list_step_ckpts(run_dir)
    if step_ckpts:
        return [(f"step{int(step):09d}", path) for step, path in step_ckpts]

    latest = run_dir / "checkpoints" / "ckpt_latest.pt"
    if latest.exists():
        return [("latest", latest)]

    return []


def run_video_suite() -> None:
    pm = normalize_policy_mode(
        DEFAULT_VIDEO_POLICY_MODE,
        allowed=("mode", "sample"),
        name="policy_mode",
    )
    device = "cpu"

    root = RUNS_ROOT.resolve()
    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    run_dirs = _discover_run_dirs(root)
    if not run_dirs:
        typer.echo(f"[suite] No run directories with checkpoints under {root}")
        return

    seeds = _normalize_seeds(DEFAULT_VIDEO_SEEDS)

    videos_root = (RESULTS_DIR / "videos").resolve()
    videos_root.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"[suite] Rendering rollout videos for {len(run_dirs)} run(s) "
        f"({pm}, eval_seeds={seeds}) to {videos_root}"
    )

    for rd in run_dirs:
        ckpts = _ckpts_for_run(rd)
        if not ckpts:
            typer.echo(f"[suite]    - {rd.name}: no checkpoints found, skipping")
            continue

        out_dir = videos_root / rd.name
        out_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"[suite]    - {rd.name}: {len(ckpts)} checkpoint(s)")

        for tag, ckpt_path in ckpts:
            for s in seeds:
                out_name = f"{tag}__{pm}__evalseed{int(s)}.mp4"
                out_path = out_dir / out_name

                if out_path.exists():
                    continue

                typer.echo(f"[suite]        * {ckpt_path.name} -> {out_path.name}")
                try:
                    render_rollout_video(
                        ckpt_path=ckpt_path,
                        out_path=out_path,
                        seed=int(s),
                        max_steps=int(DEFAULT_VIDEO_MAX_STEPS),
                        device=str(device),
                        policy_mode=pm,
                        fps=int(DEFAULT_VIDEO_FPS),
                    )
                except Exception as exc:
                    typer.echo(f"[warn] Failed to render {ckpt_path}: {exc}")
