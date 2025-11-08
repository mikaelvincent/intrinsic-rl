"""Sweep / multi-seed evaluation, aggregation, and statistics.

This CLI discovers checkpoints and evaluates them deterministically (eval-many),
writes aggregated CSVs, and (new) provides non-parametric statistical tests
between methods using the per-seed raw summary (Mann–Whitney U + bootstrap CIs).

Usage examples
--------------
# Evaluate latest checkpoints and write summary CSVs
python -m irl.sweep eval-many --runs "runs/proposed__BipedalWalker*" --out results/summary.csv

# Compare two methods on a given env using the raw results
python -m irl.sweep stats \
  --summary-raw results/summary_raw.csv \
  --env BipedalWalker-v3 \
  --method-a proposed \
  --method-b ride \
  --metric mean_return \
  --boot 5000
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
from irl.stats_utils import bootstrap_ci, mannwhitney_u  # NEW: stats helpers

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


# ---------------------- NEW: Statistics CLI ----------------------


def _read_summary_raw(path: Path) -> list[dict]:
    """Read results/summary_raw.csv into a list of dicts with typed fields."""
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append(
                    {
                        "method": str(row["method"]),
                        "env_id": str(row["env_id"]),
                        "seed": int(row["seed"]),
                        "ckpt_step": int(row["ckpt_step"]),
                        "episodes": int(row["episodes"]),
                        "mean_return": float(row["mean_return"]),
                        "std_return": float(row["std_return"]),
                        "min_return": float(row["min_return"]),
                        "max_return": float(row["max_return"]),
                        "mean_length": float(row["mean_length"]),
                        "std_length": float(row["std_length"]),
                        "ckpt_path": str(row.get("ckpt_path", "")),
                    }
                )
            except Exception:
                # Skip malformed rows; keep going
                continue
    return rows


def _values_for_method(
    raw: list[dict],
    *,
    env: str,
    method: str,
    metric: str,
    latest_per_seed: bool = True,
) -> list[float]:
    """Collect per-seed values for (env, method), choosing latest step per seed if enabled."""
    filt = [r for r in raw if r["env_id"] == env and r["method"] == method]
    if not filt:
        return []

    if latest_per_seed:
        # Keep highest ckpt_step per seed
        by_seed: dict[int, dict] = {}
        for r in filt:
            sid = int(r["seed"])
            prev = by_seed.get(sid)
            if prev is None or int(r["ckpt_step"]) > int(prev["ckpt_step"]):
                by_seed[sid] = r
        vals = [float(rec[metric]) for rec in by_seed.values()]
    else:
        vals = [float(rec[metric]) for rec in filt]

    return vals


@app.command("stats")
def cli_stats(
    summary_raw: Path = typer.Option(
        Path("results/summary_raw.csv"),
        "--summary-raw",
        "-s",
        help="Path to per-checkpoint CSV (from eval-many).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    env: str = typer.Option(
        ..., "--env", "-e", help="Environment id to filter (e.g., BipedalWalker-v3)."
    ),
    method_a: str = typer.Option(
        ..., "--method-a", "-a", help="First method name (e.g., proposed)."
    ),
    method_b: str = typer.Option(..., "--method-b", "-b", help="Second method name (e.g., ride)."),
    metric: str = typer.Option(
        "mean_return",
        "--metric",
        "-m",
        help="Column from summary_raw to compare (default: mean_return).",
    ),
    boot: int = typer.Option(
        2000, "--boot", "-B", help="Bootstrap draws for CIs (0 disables bootstrap)."
    ),
    alternative: str = typer.Option(
        "two-sided",
        "--alt",
        help='Alternative hypothesis: "two-sided" | "greater" | "less". '
        '"greater" tests if method-a tends larger than method-b.',
    ),
    latest_per_seed: bool = typer.Option(
        True,
        "--latest-per-seed/--all-checkpoints",
        help="Use the latest ckpt per seed (default) or all rows.",
    ),
) -> None:
    """Compare two methods non-parametrically with Mann–Whitney U and bootstrap CIs."""
    alt = alternative.strip().lower()
    if alt not in ("two-sided", "greater", "less"):
        raise typer.BadParameter("--alt must be one of: two-sided, greater, less")

    raw = _read_summary_raw(summary_raw)
    if not raw:
        raise typer.BadParameter(f"No rows parsed from {summary_raw}")

    x = _values_for_method(
        raw, env=env, method=method_a, metric=metric, latest_per_seed=latest_per_seed
    )
    y = _values_for_method(
        raw, env=env, method=method_b, metric=metric, latest_per_seed=latest_per_seed
    )

    if not x:
        raise typer.BadParameter(f"No rows for env={env!r}, method={method_a!r}")
    if not y:
        raise typer.BadParameter(f"No rows for env={env!r}, method={method_b!r}")

    res = mannwhitney_u(x, y, alternative=alt)  # MWU (normal approx + tie corr)

    # Bootstrap CIs for differences in mean and median
    def diff_mean(a, b):  # noqa: ANN001 - simple inline for bootstrap
        return float(float(sum(a)) / len(a) - float(sum(b)) / len(b))

    def diff_median(a, b):  # noqa: ANN001
        return float(np.median(a) - np.median(b))

    mean_pt, mean_lo, mean_hi = (
        bootstrap_ci(x, y, diff_mean, n_boot=int(boot))
        if boot > 0
        else (res.mean_x - res.mean_y, float("nan"), float("nan"))
    )
    med_pt, med_lo, med_hi = (
        bootstrap_ci(x, y, diff_median, n_boot=int(boot))
        if boot > 0
        else (res.median_x - res.median_y, float("nan"), float("nan"))
    )

    # Pretty report
    typer.echo(f"\n[bold]Mann–Whitney U test[/bold] on {env} — metric: {metric}")
    typer.echo(f"Groups: {method_a} (n={res.n_x}) vs {method_b} (n={res.n_y})")
    typer.echo(
        f"U1={res.U1:.3f}, U2={res.U2:.3f}, U_used={res.U:.3f}, z={res.z:.3f}, p={res.p_value:.6g}  (alt={alt})"
    )
    typer.echo(
        f"Means   : {method_a}={res.mean_x:.3f}, {method_b}={res.mean_y:.3f}  (Δ={res.mean_x - res.mean_y:+.3f})"
    )
    typer.echo(
        f"Medians : {method_a}={res.median_x:.3f}, {method_b}={res.median_y:.3f}  (Δ={res.median_x - res.median_y:+.3f})"
    )
    typer.echo(
        f"Effect sizes: CLES={res.cles:.3f}  Cliff's δ={res.cliffs_delta:+.3f}  (δ=2*CLES-1)"
    )

    if boot > 0:
        typer.echo(f"\nBootstrap {boot}× percentile CIs (two-sided 95%):")
        typer.echo(f"  Δ mean   : {mean_pt:+.3f}  CI [{mean_lo:+.3f}, {mean_hi:+.3f}]")
        typer.echo(f"  Δ median : {med_pt:+.3f}  CI [{med_lo:+.3f}, {med_hi:+.3f}]")

    typer.echo(
        "\nNotes:\n"
        "  • MWU p-value uses normal approximation with tie correction and continuity correction.\n"
        "  • CLES = P(a > b) + 0.5·P(a = b); Cliff's δ = 2·CLES − 1 (rank-biserial).\n"
        "  • For robust comparisons, prefer ≥5 seeds per method."
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
