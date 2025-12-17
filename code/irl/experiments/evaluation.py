from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping

import typer

from irl.cli.common import validate_policy_mode
from irl.evaluator import evaluate
from irl.pipelines.discovery import discover_run_dirs_with_latest_ckpt
from irl.pipelines.eval import evaluate_ckpt_to_run_result
from irl.results.summary import RunResult, _aggregate, _write_raw_csv, _write_summary_csv
from irl.utils.checkpoint import atomic_replace, load_checkpoint
from irl.utils.runs import parse_run_name


def _cfg_fields(payload: Mapping[str, Any]) -> tuple[str | None, str | None, int | None]:
    cfg = payload.get("cfg") or {}
    if not isinstance(cfg, Mapping):
        return None, None, None

    env_id = None
    env_cfg = cfg.get("env") or {}
    if isinstance(env_cfg, Mapping) and env_cfg.get("id") is not None:
        env_id = str(env_cfg.get("id"))

    method = str(cfg.get("method")) if cfg.get("method") is not None else None

    seed = None
    if cfg.get("seed") is not None:
        try:
            seed = int(cfg.get("seed"))
        except Exception:
            seed = None

    return env_id, method, seed


_COVERAGE_COLS: list[str] = [
    "env_id",
    "method",
    "n_runs",
    "n_seeds",
    "seeds",
    "missing_seeds",
    "ckpt_step_min",
    "ckpt_step_max",
    "ckpt_step_mean",
    "ckpt_step_median",
    "ckpt_step_std",
]


def _write_coverage_csv(rows: list[dict[str, object]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_COVERAGE_COLS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _COVERAGE_COLS})
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except Exception:
            pass

    atomic_replace(tmp, path)


def _coverage_from_results(
    results: list[RunResult],
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, set[int]]],
    dict[str, dict[str, list[int]]],
]:
    by_env_method: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    seeds_by_env: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    steps_by_env: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        env_id = str(r.env_id)
        method = str(r.method)
        by_env_method[(env_id, method)].append(r)
        seeds_by_env[env_id][method].add(int(r.seed))
        steps_by_env[env_id][method].append(int(r.ckpt_step))

    union_seeds_by_env: dict[str, set[int]] = {}
    for env_id, m_map in seeds_by_env.items():
        u: set[int] = set()
        for ss in m_map.values():
            u |= set(ss)
        union_seeds_by_env[env_id] = u

    rows: list[dict[str, object]] = []

    for (env_id, method), rs in sorted(by_env_method.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        seeds = sorted({int(r.seed) for r in rs})
        steps_all = [int(r.ckpt_step) for r in rs]
        steps = [s for s in steps_all if s >= 0]

        union = union_seeds_by_env.get(env_id, set())
        missing = sorted(set(union) - set(seeds))

        if steps:
            step_min = min(steps)
            step_max = max(steps)
            step_mean = int(round(mean(steps)))
            step_median = int(round(median(steps)))
            step_std = float(pstdev(steps)) if len(steps) > 1 else 0.0
        else:
            step_min = -1
            step_max = -1
            step_mean = -1
            step_median = -1
            step_std = 0.0

        rows.append(
            {
                "env_id": env_id,
                "method": method,
                "n_runs": int(len(rs)),
                "n_seeds": int(len(seeds)),
                "seeds": ",".join(str(s) for s in seeds),
                "missing_seeds": ",".join(str(s) for s in missing),
                "ckpt_step_min": int(step_min),
                "ckpt_step_max": int(step_max),
                "ckpt_step_mean": int(step_mean),
                "ckpt_step_median": int(step_median),
                "ckpt_step_std": float(step_std),
            }
        )

    return rows, seeds_by_env, steps_by_env


def _coverage_msg_seed(env_id: str, missing_by_method: dict[str, list[int]]) -> str:
    bits: list[str] = []
    for m, miss in sorted(missing_by_method.items(), key=lambda kv: kv[0]):
        if miss:
            bits.append(f"{m} missing {miss}")
    return f"[suite]    ! Seed coverage mismatch for env={env_id}: " + "; ".join(bits)


def _coverage_msg_step(env_id: str, step_mean_by_method: dict[str, int]) -> str:
    bits = [f"{m}≈{s}" for m, s in sorted(step_mean_by_method.items(), key=lambda kv: kv[0])]
    return f"[suite]    ! Step parity mismatch for env={env_id}: " + ", ".join(bits)


def _enforce_coverage_and_step_parity(
    seeds_by_env: Mapping[str, Mapping[str, set[int]]],
    steps_by_env: Mapping[str, Mapping[str, list[int]]],
    *,
    strict_coverage: bool,
    strict_step_parity: bool,
) -> None:
    for env_id in sorted(seeds_by_env.keys()):
        methods = seeds_by_env[env_id]
        if len(methods) <= 1:
            continue

        union: set[int] = set()
        for ss in methods.values():
            union |= set(ss)

        missing_by_method: dict[str, list[int]] = {}
        for m, ss in methods.items():
            miss = sorted(union - set(ss))
            if miss:
                missing_by_method[str(m)] = miss

        if missing_by_method:
            msg = _coverage_msg_seed(str(env_id), missing_by_method)
            if strict_coverage:
                raise RuntimeError(msg)
            typer.echo(msg)

        step_means: dict[str, int] = {}
        for m, steps in steps_by_env.get(env_id, {}).items():
            steps_ok = [int(s) for s in steps if int(s) >= 0]
            if not steps_ok:
                continue
            step_means[str(m)] = int(round(mean(steps_ok)))

        if len(step_means) <= 1:
            continue

        max_step = max(step_means.values())
        min_step = min(step_means.values())
        thresh = max(1000, int(round(0.05 * max_step)))
        if (max_step - min_step) > thresh:
            msg = _coverage_msg_step(str(env_id), step_means)
            if strict_step_parity:
                raise RuntimeError(msg)
            typer.echo(msg)


def _discover_run_dirs_with_ckpt(runs_root: Path) -> list[tuple[Path, Path]]:
    return discover_run_dirs_with_latest_ckpt(runs_root)


def run_eval_suite(
    runs_root: Path,
    results_dir: Path,
    episodes: int,
    device: str,
    policy_mode: str = "mode",
    *,
    strict_coverage: bool = True,
    strict_step_parity: bool = True,
) -> None:
    pm = validate_policy_mode(policy_mode, allowed=("mode", "sample"))

    root = runs_root.resolve()
    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    runs = _discover_run_dirs_with_ckpt(root)
    if not runs:
        typer.echo(f"[suite] No run directories with checkpoints under {root}")
        return

    typer.echo(f"[suite] Evaluating {len(runs)} run(s) from {root}")
    results: list[RunResult] = []

    traj_root = results_dir / "plots" / "trajectories"
    traj_root.mkdir(parents=True, exist_ok=True)

    for rd, ckpt in runs:
        try:
            payload = load_checkpoint(ckpt, map_location="cpu")

            cfg_env_id, cfg_method, cfg_seed = _cfg_fields(payload)

            info = parse_run_name(rd)
            run_env_tag = info.get("env")
            run_method = info.get("method")
            run_seed = None
            try:
                if info.get("seed") is not None:
                    run_seed = int(info.get("seed"))
            except Exception:
                run_seed = None

            if cfg_env_id is None:
                typer.echo(
                    f"[suite]    ! {rd.name}: cfg.env.id missing in checkpoint; "
                    f"falling back to run dir label {run_env_tag!r}."
                )

            env_for_eval = cfg_env_id or str(run_env_tag or "UnknownEnv")

            if cfg_env_id is not None and run_env_tag is not None:
                expected_tag = str(cfg_env_id).replace("/", "-")
                if str(run_env_tag) != expected_tag:
                    typer.echo(
                        f"[suite]    ! {rd.name}: run dir env tag {run_env_tag!r} "
                        f"!= cfg.env.id {cfg_env_id!r}; using cfg.env.id."
                    )

            if cfg_method is not None and run_method is not None:
                if str(cfg_method).strip().lower() != str(run_method).strip().lower():
                    typer.echo(
                        f"[suite]    ! {rd.name}: run dir method {run_method!r} "
                        f"!= cfg.method {cfg_method!r}; using cfg.method."
                    )

            if cfg_seed is not None and run_seed is not None and int(cfg_seed) != int(run_seed):
                typer.echo(
                    f"[suite]    ! {rd.name}: run dir seed {run_seed} "
                    f"!= cfg.seed {cfg_seed}; using cfg.seed."
                )

            typer.echo(
                f"[suite]    - {rd.name}: ckpt={ckpt.name}, env={env_for_eval}, episodes={episodes}"
            )

            traj_out_dir = traj_root / rd.name
            traj_out_dir.mkdir(parents=True, exist_ok=True)

            rr = evaluate_ckpt_to_run_result(
                ckpt,
                payload=payload,
                env=str(env_for_eval),
                method=(cfg_method or run_method or "unknown"),
                seed=int(cfg_seed) if cfg_seed is not None else None,
                episodes=int(episodes),
                device=str(device),
                policy_mode=pm,
                save_traj=True,
                traj_out_dir=traj_out_dir,
                evaluate_fn=evaluate,
            )
            results.append(rr)
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

    coverage_rows, seeds_by_env, steps_by_env = _coverage_from_results(results)
    coverage_path = results_root / "coverage.csv"
    _write_coverage_csv(coverage_rows, coverage_path)
    typer.echo(f"[suite] Wrote coverage report to {coverage_path}")

    _enforce_coverage_and_step_parity(
        seeds_by_env,
        steps_by_env,
        strict_coverage=bool(strict_coverage),
        strict_step_parity=bool(strict_step_parity),
    )
