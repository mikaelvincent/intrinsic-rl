from __future__ import annotations

from pathlib import Path

import typer

from irl.evaluator import evaluate
from irl.plot import _parse_run_name
from irl.sweep import RunResult, _aggregate, _find_latest_ckpt, _write_raw_csv, _write_summary_csv
from irl.utils.checkpoint import load_checkpoint


def run_eval_suite(runs_root: Path, results_dir: Path, episodes: int, device: str) -> None:
    root = runs_root.resolve()
    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not run_dirs:
        typer.echo(f"[suite] No run directories under {root}")
        return

    typer.echo(f"[suite] Evaluating {len(run_dirs)} run(s) from {root}")
    results: list[RunResult] = []

    traj_root = results_dir / "plots" / "trajectories"
    traj_root.mkdir(parents=True, exist_ok=True)

    for rd in run_dirs:
        ckpt = _find_latest_ckpt(rd)
        if ckpt is None:
            typer.echo(f"[suite]    - {rd.name}: no checkpoints found, skipping")
            continue
        typer.echo(f"[suite]    - {rd.name}: ckpt={ckpt.name}, episodes={episodes}")
        try:
            traj_out_dir = traj_root / rd.name
            traj_out_dir.mkdir(parents=True, exist_ok=True)

            summary = evaluate(
                env=str(_parse_run_name(rd).get("env", "UnknownEnv")),
                ckpt=ckpt,
                episodes=episodes,
                device=device,
                save_traj=True,
                traj_out_dir=traj_out_dir,
            )

            payload = load_checkpoint(ckpt, map_location="cpu")
            step = int(payload.get("step", -1))
            info = _parse_run_name(rd)

            results.append(
                RunResult(
                    method=info.get("method", "unknown"),
                    env_id=summary["env_id"],
                    seed=int(info.get("seed", 0)),
                    ckpt_path=ckpt,
                    ckpt_step=step,
                    episodes=int(summary["episodes"]),
                    mean_return=float(summary["mean_return"]),
                    std_return=float(summary["std_return"]),
                    min_return=float(summary["min_return"]),
                    max_return=float(summary["max_return"]),
                    mean_length=float(summary["mean_length"]),
                    std_length=float(summary["std_length"]),
                )
            )
        except Exception as exc:
            typer.echo(f"[suite]          ! evaluation failed: {exc}")

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
