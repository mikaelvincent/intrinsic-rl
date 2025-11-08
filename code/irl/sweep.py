"""Sweep / multi-seed evaluation and aggregation.

This CLI discovers one or more checkpoints (either passed directly or found
under run directories), evaluates each deterministically using the existing
`irl.evaluator.evaluate` routine, and writes aggregated CSV summaries.

Usage examples
--------------
# Evaluate latest checkpoints under several run directories (glob accepted)
python -m irl.sweep eval-many \
  --runs "runs/proposed__BipedalWalker*" \
  --episodes 10 \
  --device cpu \
  --out results/summary.csv

# Evaluate an explicit list of checkpoints (mixing methods/envs allowed)
python -m irl.sweep eval-many \
  --ckpt runs/.../checkpoints/ckpt_step_100000.pt \
  --ckpt runs/.../checkpoints/ckpt_latest.pt \
  --out results/summary.csv

Outputs
-------
* summary.csv        — aggregated by (method, env_id) across seeds
* summary_raw.csv    — per-checkpoint (per seed) raw results (same folder)
"""

from __future__ import annotations

import csv
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

import typer

from irl.evaluator import evaluate
from irl.utils.checkpoint import load_checkpoint

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")

# Local regex to identify step-numbered checkpoints
_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


@dataclass
class RunResult:
    """Per-checkpoint (per seed) evaluation record."""

    method: str
    env_id: str
    seed: int
    ckpt_path: Path
    ckpt_step: int
    episodes: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    mean_length: float
    std_length: float


def _find_latest_ckpt(run_dir: Path) -> Optional[Path]:
    """Return the latest checkpoint Path inside a run directory, if any."""
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    latest = ckpt_dir / "ckpt_latest.pt"
    if latest.exists():
        return latest

    # Fall back to highest step filename
    candidates: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        if not p.is_file():
            continue
        m = _CKPT_RE.match(p.name)
        if not m:
            continue
        try:
            step = int(m.group(1))
            candidates.append((step, p))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _collect_ckpts_from_runs(run_globs: Iterable[str]) -> List[Path]:
    """Expand globs and return latest checkpoints from each matching run dir."""
    out: list[Path] = []
    for pattern in run_globs:
        for p in glob.glob(pattern):
            rd = Path(p)
            if rd.is_file():
                # Allow passing a checkpoint file via --runs by accident; accept it.
                if rd.name.startswith("ckpt_") and rd.suffix == ".pt":
                    out.append(rd.resolve())
                continue
            ck = _find_latest_ckpt(rd)
            if ck is not None:
                out.append(ck.resolve())
    # De-duplicate while preserving order
    seen = set()
    unique: list[Path] = []
    for p in out:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _normalize_inputs(runs: Optional[List[str]], ckpts: Optional[List[Path]]) -> List[Path]:
    """Union of checkpoints gathered from run globs and explicit --ckpt paths."""
    paths: list[Path] = []
    if runs:
        paths.extend(_collect_ckpts_from_runs(runs))
    if ckpts:
        for c in ckpts:
            if c.exists() and c.is_file():
                paths.append(c.resolve())
    # De-duplicate
    uniq: list[Path] = []
    seen = set()
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _evaluate_ckpt(ckpt: Path, episodes: int, device: str) -> RunResult:
    """Load metadata from the checkpoint, pick env/method/seed, and evaluate."""
    payload = load_checkpoint(ckpt, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    env_id = str(((cfg.get("env") or {}).get("id")) or "MountainCar-v0")
    method = str(cfg.get("method", "vanilla"))
    seed = int(cfg.get("seed", 1))
    step = int(payload.get("step", -1))

    summary = evaluate(env=env_id, ckpt=ckpt, episodes=episodes, device=device)

    return RunResult(
        method=method,
        env_id=summary["env_id"],
        seed=int(summary["seed"]),
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


def _write_raw_csv(rows: List[RunResult], path: Path) -> None:
    """Write per-checkpoint results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fn = path
    with fn.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "env_id",
                "seed",
                "ckpt_step",
                "episodes",
                "mean_return",
                "std_return",
                "min_return",
                "max_return",
                "mean_length",
                "std_length",
                "ckpt_path",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.method,
                    r.env_id,
                    r.seed,
                    r.ckpt_step,
                    r.episodes,
                    f"{r.mean_return:.6f}",
                    f"{r.std_return:.6f}",
                    f"{r.min_return:.6f}",
                    f"{r.max_return:.6f}",
                    f"{r.mean_length:.6f}",
                    f"{r.std_length:.6f}",
                    str(r.ckpt_path),
                ]
            )


def _aggregate(rows: List[RunResult]) -> List[Dict[str, object]]:
    """Aggregate per-seed rows by (method, env_id)."""
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for r in rows:
        key = (r.method, r.env_id)
        groups.setdefault(key, []).append(r)

    out: list[dict[str, object]] = []
    for (method, env_id), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        seeds = sorted({int(r.seed) for r in rs})
        n = len(seeds)

        means = [r.mean_return for r in rs]
        lens_means = [r.mean_length for r in rs]
        steps = [int(r.ckpt_step) for r in rs]

        agg = {
            "method": method,
            "env_id": env_id,
            "episodes_per_seed": int(rs[0].episodes) if rs else 0,
            "n_seeds": n,
            "seeds": ",".join(str(s) for s in seeds),
            "mean_return_mean": float(mean(means)) if n > 0 else 0.0,
            "mean_return_std": float(pstdev(means)) if n > 1 else 0.0,
            "mean_length_mean": float(mean(lens_means)) if n > 0 else 0.0,
            "mean_length_std": float(pstdev(lens_means)) if n > 1 else 0.0,
            "step_min": min(steps) if steps else -1,
            "step_max": max(steps) if steps else -1,
            "step_mean": int(round(mean(steps))) if steps else -1,
        }
        out.append(agg)
    return out


def _write_summary_csv(agg_rows: List[Dict[str, object]], path: Path) -> None:
    """Write aggregated summary to CSV (one row per (method, env))."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "method",
        "env_id",
        "episodes_per_seed",
        "n_seeds",
        "seeds",
        "mean_return_mean",
        "mean_return_std",
        "mean_length_mean",
        "mean_length_std",
        "step_min",
        "step_max",
        "step_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in agg_rows:
            w.writerow(row)


@app.command("eval-many")
def cli_eval_many(
    runs: Optional[List[str]] = typer.Option(
        None,
        "--runs",
        "-r",
        help='Glob(s) to run directories (e.g., "runs/proposed__BipedalWalker*").',
    ),
    ckpt: Optional[List[Path]] = typer.Option(
        None,
        "--ckpt",
        "-k",
        help="Explicit checkpoint path(s) to evaluate (can be repeated).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    episodes: int = typer.Option(10, "--episodes", "-n", help="Episodes per checkpoint."),
    device: str = typer.Option(
        "cpu", "--device", "-d", help='Torch device, e.g. "cpu" or "cuda:0".'
    ),
    out: Path = typer.Option(
        Path("results/summary.csv"),
        "--out",
        "-o",
        help="Path to aggregated CSV (summary.csv). A sibling summary_raw.csv is also written.",
        dir_okay=False,
    ),
) -> None:
    """Evaluate multiple checkpoints (multi-seed) and export CSV summaries."""
    ckpts = _normalize_inputs(runs, ckpt)
    if not ckpts:
        raise typer.BadParameter("No checkpoints found. Provide --runs and/or --ckpt.")

    typer.echo(f"[info] Found {len(ckpts)} checkpoint(s). Starting evaluation...")

    # Evaluate each checkpoint deterministically
    results: list[RunResult] = []
    for i, c in enumerate(ckpts, start=1):
        typer.echo(f"  [{i}/{len(ckpts)}] {c}")
        try:
            res = _evaluate_ckpt(c, episodes=episodes, device=device)
            results.append(res)
        except Exception as exc:
            typer.echo(f"[warning] Failed to evaluate {c}: {exc}")

    if not results:
        raise typer.Exit(code=1)

    # Write raw per-checkpoint results
    raw_path = out.parent / "summary_raw.csv"
    _write_raw_csv(results, raw_path)

    # Aggregate and write summary.csv
    agg = _aggregate(results)
    _write_summary_csv(agg, out)

    # Short console report
    typer.echo("\n[green]Aggregation complete[/green]")
    typer.echo(f"Per-checkpoint results: {raw_path}")
    typer.echo(f"Aggregated summary    : {out}")
    typer.echo("\nTop-lines:")
    for row in agg:
        typer.echo(
            f"  {row['method']:>8} | {row['env_id']:<18} | seeds={row['seeds']:<10} "
            f"| mean={row['mean_return_mean']:.2f} ± {row['mean_return_std']:.2f} "
            f"(step≈{row['step_mean']})"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
