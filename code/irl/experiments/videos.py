"""Video-generation helpers for the experiment suite.

This suite generates side-by-side comparison videos (Baseline vs Proposed) for
*every checkpoint* in the selected runs.

Selection policy
----------------
For each environment:

1. Choose one run directory per method:
   * If `results_dir/summary_raw.csv` exists, pick the run corresponding to the
     best (highest mean_return) checkpoint row for that `(env_id, method)`.
   * Otherwise, fall back to the first matching run directory under `runs_root`.

2. Enumerate all step-numbered checkpoints under each selected run:
   `checkpoints/ckpt_step_<N>.pt`.

3. Render one video per checkpoint step `N`, pairing each side with the latest
   checkpoint *at or before* `N` (so mismatched checkpoint cadences still work).

Outputs are stored under:

    <results_dir>/videos/<env_tag>/<env_tag>__<baseline>_vs_<method>__step#########.mp4
"""

from __future__ import annotations

import re
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import typer

from irl.plot import _parse_run_name
from irl.video import render_side_by_side

_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


def _run_dir_from_ckpt_path(p: Path) -> Optional[Path]:
    """Best-effort resolve the run directory from a checkpoint path."""
    try:
        p = Path(p)
    except Exception:
        return None

    if p.is_dir():
        # Already a run directory (uncommon, but tolerate).
        return p

    # Typical layout: <run_dir>/checkpoints/ckpt_*.pt
    if p.is_file():
        if p.parent.name == "checkpoints":
            return p.parent.parent
        return p.parent

    # If it's a stringy path that doesn't exist, still infer by structure.
    # (Useful for tests that touch files but may not fully mirror real structure.)
    if p.name.startswith("ckpt_") and p.parent.name == "checkpoints":
        return p.parent.parent

    return None


def _list_step_checkpoints(run_dir: Path) -> Dict[int, Path]:
    """Return {step: ckpt_path} for ckpt_step_<step>.pt files in run_dir/checkpoints."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return {}

    out: Dict[int, Path] = {}
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


def _latest_alias(run_dir: Path) -> Optional[Path]:
    """Return ckpt_latest.pt if present."""
    p = Path(run_dir) / "checkpoints" / "ckpt_latest.pt"
    return p if p.exists() else None


def _select_at_or_before(steps_sorted: list[int], ckpts: Dict[int, Path], target_step: int) -> Path:
    """Pick the checkpoint with the largest step <= target_step, else the earliest."""
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
    method: str = "proposed",
) -> None:
    """Generate side-by-side videos: Baseline vs Proposed at every checkpoint."""
    root = runs_root.resolve()
    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    # Run directories are expected to be immediate children (suite layout).
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

    # Best-run selection via summary_raw.csv (if present).
    # We store run_dir (not ckpt_path) so we can enumerate all its checkpoints.
    summary_csv = Path(results_dir) / "summary_raw.csv"
    best_run_dirs: Dict[Tuple[str, str], Path] = {}
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
            # Best-effort only; fall back to directory scan if parsing fails.
            best_run_dirs = {}

    # Discover envs from run directory names
    envs: Set[str] = set()
    for rd in run_dirs:
        info = _parse_run_name(rd)
        if "env" in info:
            envs.add(info["env"])

    if not envs:
        typer.echo("[suite] No environments found.")
        return

    videos_root = Path(results_dir) / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"[suite] Generating checkpoint videos ({baseline} vs {method}) for {len(envs)} envs..."
    )

    for env_id in sorted(envs):
        env_tag = env_id.replace("/", "-")

        # --- Resolve baseline run dir ---
        rd_left = best_run_dirs.get((env_id, baseline))
        if rd_left is None:
            matches = [
                rd
                for rd in run_dirs
                if _parse_run_name(rd).get("env") == env_id
                and _parse_run_name(rd).get("method") == baseline
            ]
            rd_left = matches[0] if matches else None

        # --- Resolve method run dir ---
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
            typer.echo(
                f"[suite]    - {env_id}: missing run dirs for one or both methods. Skipping."
            )
            continue

        # Enumerate checkpoints
        ckpts_left = _list_step_checkpoints(rd_left)
        ckpts_right = _list_step_checkpoints(rd_right)
        latest_left = _latest_alias(rd_left)
        latest_right = _latest_alias(rd_right)

        # Steps we will render videos for: union across both sides.
        steps_union = sorted(set(ckpts_left.keys()) | set(ckpts_right.keys()))

        # If neither side has step-numbered checkpoints, fall back to a single "latest" render.
        if not steps_union:
            if latest_left is None or latest_right is None:
                typer.echo(
                    f"[suite]    - {env_id}: no step checkpoints and missing ckpt_latest.pt. Skipping."
                )
                continue
            steps_to_render = ["latest"]
        else:
            steps_to_render = steps_union  # type: ignore[assignment]

        out_dir = videos_root / env_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(
            f"[suite]    - {env_id}: Rendering {len(steps_to_render)} checkpoint video(s) to {out_dir}..."
        )

        # Pre-sort step lists for efficient at-or-before selection.
        left_steps_sorted = sorted(ckpts_left.keys())
        right_steps_sorted = sorted(ckpts_right.keys())

        for s in steps_to_render:
            if s == "latest":
                ckpt_l = latest_left  # type: ignore[assignment]
                ckpt_r = latest_right  # type: ignore[assignment]
                assert ckpt_l is not None and ckpt_r is not None
                step_tag = "latest"
            else:
                step = int(s)

                if ckpts_left:
                    ckpt_l = _select_at_or_before(left_steps_sorted, ckpts_left, step)
                else:
                    ckpt_l = latest_left

                if ckpts_right:
                    ckpt_r = _select_at_or_before(right_steps_sorted, ckpts_right, step)
                else:
                    ckpt_r = latest_right

                if ckpt_l is None or ckpt_r is None:
                    typer.echo(
                        f"[suite]        * step={step}: missing ckpt on one side; skipping."
                    )
                    continue

                # Zero-pad for lexical ordering in file browsers.
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
                    seed=100,  # Fixed seed for visualization fairness (and determinism)
                    device=device,
                    label_left=baseline.capitalize(),
                    label_right=method.capitalize(),
                )
            except Exception as exc:
                typer.echo(f"[warn] Failed to render video for {env_id} ({step_tag}): {exc}")
