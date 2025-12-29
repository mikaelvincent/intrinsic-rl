from __future__ import annotations

from pathlib import Path

import typer

from irl.visualization.figures import plot_trajectory_heatmap


def _generate_trajectory_plots(results_dir: Path, _plots_root: Path) -> None:
    traj_dir = Path(results_dir) / "plots" / "trajectories"
    if not traj_dir.exists():
        return

    npz_files = sorted(traj_dir.rglob("*_trajectory.npz"), key=lambda p: str(p))
    if not npz_files:
        return

    typer.echo(f"[suite] Generating trajectory heatmaps for {len(npz_files)} files...")

    for npz_file in npz_files:
        env_tag = npz_file.stem.replace("_trajectory", "")
        out_path = npz_file.with_name(f"{env_tag}__state_heatmap.png")
        try:
            wrote = bool(plot_trajectory_heatmap(npz_file, out_path))
            if wrote:
                typer.echo(f"[suite] Saved heatmap: {out_path}")
            else:
                typer.echo(f"[suite] Skipped heatmap (no projection): {npz_file}")
        except Exception as exc:
            typer.echo(f"[warn] Failed to plot heatmap for {npz_file}: {exc}")
