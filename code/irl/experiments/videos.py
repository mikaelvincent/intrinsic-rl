from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import typer

from irl.cli.validators import normalize_policy_mode
from irl.pipelines.discovery import discover_run_dirs_with_step_ckpts
from irl.utils.runs import list_step_ckpts
from irl.video import render_rollout_video


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
    pm = normalize_policy_mode(policy_mode, allowed=("mode", "sample"), name="policy_mode")

    root = Path(runs_root).resolve()
    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    run_dirs = discover_run_dirs_with_step_ckpts(root)
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
        ckpts = list_step_ckpts(rd)
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
