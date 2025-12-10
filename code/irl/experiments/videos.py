"""Video-generation helpers for the experiment suite."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import typer

from irl.plot import _parse_run_name
from irl.sweep import _find_latest_ckpt
from irl.video import render_side_by_side


def run_video_suite(
    runs_root: Path,
    results_dir: Path,
    device: str,
    baseline: str = "vanilla",
    method: str = "proposed",
) -> None:
    """Generate side-by-side videos: Baseline vs Proposed.

    Scans runs_root for both methods. Selects the run with the *highest
    aggregated return* (if summary.csv is available) or picks the first
    found seed as a fallback.
    """
    root = runs_root.resolve()
    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    # Load summary for picking best seeds
    summary_csv = results_dir / "summary_raw.csv"
    best_runs: Dict[tuple[str, str], Path] = {}
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            # Find best row per (env, method)
            # Sort by mean_return descending
            df = df.sort_values("mean_return", ascending=False)
            for _, row in df.iterrows():
                key = (str(row["env_id"]), str(row["method"]))
                if key not in best_runs:
                    best_runs[key] = Path(row["ckpt_path"])
        except Exception:
            pass

    # Discover envs
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    envs: Set[str] = set()
    for rd in run_dirs:
        info = _parse_run_name(rd)
        if "env" in info:
            envs.add(info["env"])

    if not envs:
        typer.echo("[suite] No environments found.")
        return

    video_dir = results_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[suite] Generating comparison videos ({baseline} vs {method}) for {len(envs)} envs...")

    for env_id in sorted(envs):
        # 1. Find checkpoint for Baseline
        ckpt_l = best_runs.get((env_id, baseline))
        if ckpt_l is None:
            # Fallback scan
            matches = [
                rd
                for rd in run_dirs
                if _parse_run_name(rd).get("env") == env_id
                and _parse_run_name(rd).get("method") == baseline
            ]
            if matches:
                ckpt_l = _find_latest_ckpt(matches[0])

        # 2. Find checkpoint for Proposed
        ckpt_r = best_runs.get((env_id, method))
        if ckpt_r is None:
            matches = [
                rd
                for rd in run_dirs
                if _parse_run_name(rd).get("env") == env_id
                and _parse_run_name(rd).get("method") == method
            ]
            if matches:
                ckpt_r = _find_latest_ckpt(matches[0])

        if ckpt_l is None or ckpt_r is None:
            typer.echo(f"[suite]    - {env_id}: missing checkpoints for one or both methods. Skipping.")
            continue

        out_name = f"{env_id.replace('/', '-')}__{baseline}_vs_{method}.mp4"
        out_path = video_dir / out_name

        typer.echo(f"[suite]    - {env_id}: Rendering to {out_path}...")
        try:
            render_side_by_side(
                env_id=env_id,
                ckpt_left=ckpt_l,
                ckpt_right=ckpt_r,
                out_path=out_path,
                seed=100,  # Fixed seed for visualization fairness (and determinism)
                device=device,
                label_left=baseline.capitalize(),
                label_right=method.capitalize(),
            )
        except Exception as exc:
            typer.echo(f"[warn] Failed to render video for {env_id}: {exc}")
