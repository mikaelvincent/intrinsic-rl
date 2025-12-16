from __future__ import annotations

import re
from bisect import bisect_right
from pathlib import Path

import pandas as pd
import typer

from irl.plot import _parse_run_name
from irl.video import render_side_by_side

_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


def _run_dir_from_ckpt_path(p: Path) -> Path | None:
    try:
        p = Path(p)
    except Exception:
        return None

    if p.is_dir():
        return p

    if p.is_file():
        if p.parent.name == "checkpoints":
            return p.parent.parent
        return p.parent

    if p.name.startswith("ckpt_") and p.parent.name == "checkpoints":
        return p.parent.parent

    return None


def _list_step_checkpoints(run_dir: Path) -> dict[int, Path]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return {}

    out: dict[int, Path] = {}
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
        out[step] = p
    return out


def _latest_alias(run_dir: Path) -> Path | None:
    p = Path(run_dir) / "checkpoints" / "ckpt_latest.pt"
    return p if p.exists() else None


def _select_at_or_before(steps_sorted: list[int], ckpts: dict[int, Path], target_step: int) -> Path:
    if not steps_sorted:
        raise ValueError("steps_sorted must be non-empty")
    idx = bisect_right(steps_sorted, int(target_step)) - 1
    if idx < 0:
        return ckpts[steps_sorted[0]]
    return ckpts[steps_sorted[idx]]


def run_video_suite(
    runs_root: Path,
    results_dir: Path,
    device: str,
    baseline: str = "vanilla",
    method: str = "glpe",
) -> None:
    root = runs_root.resolve()
    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

    summary_csv = Path(results_dir) / "summary_raw.csv"
    best_run_dirs: dict[tuple[str, str], Path] = {}
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            if "mean_return" in df.columns:
                df = df.sort_values("mean_return", ascending=False)
            for _, row in df.iterrows():
                env_id = str(row.get("env_id", "")).strip()
                mth = str(row.get("method", "")).strip()
                ckpt_str = str(row.get("ckpt_path", "")).strip()
                if not env_id or not mth or not ckpt_str:
                    continue
                rd = _run_dir_from_ckpt_path(Path(ckpt_str))
                if rd is None:
                    continue
                key = (env_id, mth)
                if key not in best_run_dirs:
                    best_run_dirs[key] = rd
        except Exception:
            best_run_dirs = {}

    envs: set[str] = set()
    for rd in run_dirs:
        info = _parse_run_name(rd)
        if "env" in info:
            envs.add(info["env"])

    if not envs:
        typer.echo("[suite] No environments found.")
        return

    videos_root = Path(results_dir) / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[suite] Generating checkpoint videos ({baseline} vs {method}) for {len(envs)} envs...")

    for env_id in sorted(envs):
        env_tag = env_id.replace("/", "-")

        rd_left = best_run_dirs.get((env_id, baseline))
        if rd_left is None:
            matches = [
                rd
                for rd in run_dirs
                if _parse_run_name(rd).get("env") == env_id
                and _parse_run_name(rd).get("method") == baseline
            ]
            rd_left = matches[0] if matches else None

        rd_right = best_run_dirs.get((env_id, method))
        if rd_right is None:
            matches = [
                rd
                for rd in run_dirs
                if _parse_run_name(rd).get("env") == env_id
                and _parse_run_name(rd).get("method") == method
            ]
            rd_right = matches[0] if matches else None

        if rd_left is None or rd_right is None:
            typer.echo(f"[suite]    - {env_id}: missing run dirs for one or both methods. Skipping.")
            continue

        ckpts_left = _list_step_checkpoints(rd_left)
        ckpts_right = _list_step_checkpoints(rd_right)
        latest_left = _latest_alias(rd_left)
        latest_right = _latest_alias(rd_right)

        steps_union = sorted(set(ckpts_left.keys()) | set(ckpts_right.keys()))
        if not steps_union:
            if latest_left is None or latest_right is None:
                typer.echo(
                    f"[suite]    - {env_id}: no step checkpoints and missing ckpt_latest.pt. Skipping."
                )
                continue
            steps_to_render: list[int | str] = ["latest"]
        else:
            steps_to_render = steps_union

        out_dir = videos_root / env_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(
            f"[suite]    - {env_id}: Rendering {len(steps_to_render)} checkpoint video(s) to {out_dir}..."
        )

        left_steps_sorted = sorted(ckpts_left.keys())
        right_steps_sorted = sorted(ckpts_right.keys())

        for s in steps_to_render:
            if s == "latest":
                ckpt_l = latest_left
                ckpt_r = latest_right
                assert ckpt_l is not None and ckpt_r is not None
                step_tag = "latest"
            else:
                step = int(s)
                ckpt_l = (
                    _select_at_or_before(left_steps_sorted, ckpts_left, step)
                    if ckpts_left
                    else latest_left
                )
                ckpt_r = (
                    _select_at_or_before(right_steps_sorted, ckpts_right, step)
                    if ckpts_right
                    else latest_right
                )
                if ckpt_l is None or ckpt_r is None:
                    typer.echo(f"[suite]        * step={step}: missing ckpt on one side; skipping.")
                    continue
                step_tag = f"step{step:09d}"

            out_name = f"{env_tag}__{baseline}_vs_{method}__{step_tag}.mp4"
            out_path = out_dir / out_name

            typer.echo(
                f"[suite]        * {env_id}: {step_tag} -> {out_path.name} "
                f"(L={Path(ckpt_l).name}, R={Path(ckpt_r).name})"
            )

            try:
                render_side_by_side(
                    env_id=env_id,
                    ckpt_left=Path(ckpt_l),
                    ckpt_right=Path(ckpt_r),
                    out_path=out_path,
                    seed=100,
                    device=device,
                    label_left=baseline.capitalize(),
                    label_right="Gated Learning-Progress Exploration (GLPE)",
                )
            except Exception as exc:
                typer.echo(f"[warn] Failed to render video for {env_id} ({step_tag}): {exc}")
