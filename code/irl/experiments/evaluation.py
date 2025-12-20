from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping

import typer

from irl.checkpoints.runtime import extract_eval_settings as _extract_eval_settings
from irl.cli.validators import normalize_policy_mode
from irl.evaluator import evaluate
from irl.paper_defaults import DEFAULT_EVAL_POLICY_MODE
from irl.pipelines.discovery import discover_run_dirs_with_latest_ckpt, select_ckpts_for_run
from irl.pipelines.eval import EvalCheckpoint, cfg_fields_from_payload as _cfg_fields_from_payload
from irl.pipelines.eval import evaluate_checkpoints
from irl.results.summary import (
    RunResult,
    aggregate_results,
    aggregate_results_by_step,
    write_raw_csv,
    write_summary_by_step_csv,
    write_summary_csv,
)
from irl.utils.checkpoint import atomic_write_text, load_checkpoint
from irl.utils.io import atomic_write_csv
from irl.utils.runs import parse_run_name


def _cfg_fields(payload: Mapping[str, Any]) -> tuple[str | None, str | None, int | None]:
    return _cfg_fields_from_payload(payload)


def _eval_interval_steps(payload: Mapping[str, Any]) -> int:
    return int(_extract_eval_settings(payload).interval_steps)


def _eval_episodes(payload: Mapping[str, Any]) -> int:
    return int(_extract_eval_settings(payload).episodes)


def _eval_device(payload: Mapping[str, Any]) -> str:
    return str(_extract_eval_settings(payload).device)


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
    "latest_ckpt_step_min",
    "latest_ckpt_step_max",
    "latest_ckpt_step_mean",
    "latest_ckpt_step_median",
    "latest_ckpt_step_std",
]


def _write_coverage_csv(rows: list[dict[str, object]], path: Path) -> None:
    atomic_write_csv(
        path,
        _COVERAGE_COLS,
        ({k: r.get(k, "") for k in _COVERAGE_COLS} for r in rows),
    )


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

        latest_by_seed: dict[int, int] = {}
        for r in rs:
            s = int(r.ckpt_step)
            if s < 0:
                continue
            sid = int(r.seed)
            prev = latest_by_seed.get(sid)
            if prev is None or s > prev:
                latest_by_seed[sid] = s

        latest_steps = [int(s) for s in latest_by_seed.values() if int(s) >= 0]
        if latest_steps:
            latest_min = min(latest_steps)
            latest_max = max(latest_steps)
            latest_mean = int(round(mean(latest_steps)))
            latest_median = int(round(median(latest_steps)))
            latest_std = float(pstdev(latest_steps)) if len(latest_steps) > 1 else 0.0
        else:
            latest_min = -1
            latest_max = -1
            latest_mean = -1
            latest_median = -1
            latest_std = 0.0

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
                "latest_ckpt_step_min": int(latest_min),
                "latest_ckpt_step_max": int(latest_max),
                "latest_ckpt_step_mean": int(latest_mean),
                "latest_ckpt_step_median": int(latest_median),
                "latest_ckpt_step_std": float(latest_std),
            }
        )

    return rows, seeds_by_env, steps_by_env


def _coverage_msg_seed(env_id: str, missing_by_method: dict[str, list[int]]) -> str:
    bits: list[str] = []
    for m, miss in sorted(missing_by_method.items(), key=lambda kv: kv[0]):
        if miss:
            bits.append(f"{m} missing {miss}")
    return f"[suite]    ! Seed coverage mismatch for env={env_id}: " + "; ".join(bits)


def _fmt_int_list(xs: list[int], *, limit: int = 8) -> str:
    if not xs:
        return "[]"
    if len(xs) <= limit:
        return str(xs)
    head = xs[:limit]
    return "[" + ", ".join(str(x) for x in head) + ", ...]"


def _coverage_msg_latest_step(env_id: str, seed: int, latest_by_method: Mapping[str, int]) -> str:
    bits = [f"{m}={int(s)}" for m, s in sorted(latest_by_method.items(), key=lambda kv: kv[0])]
    return f"[suite]    ! Step parity mismatch for env={env_id} seed={int(seed)}: " + ", ".join(bits)


def _coverage_msg_step_sets(env_id: str, union_steps: set[int], missing_by_method: Mapping[str, list[int]]) -> str:
    union_sorted = sorted(int(s) for s in union_steps if int(s) >= 0)
    if union_sorted:
        union_desc = f"{union_sorted[0]}..{union_sorted[-1]} (n={len(union_sorted)})"
    else:
        union_desc = "empty"

    bits: list[str] = []
    for m, miss in sorted(missing_by_method.items(), key=lambda kv: kv[0]):
        if miss:
            bits.append(f"{m} missing {_fmt_int_list(miss)}")
    return (
        f"[suite]    ! Step parity mismatch for env={env_id}: "
        f"evaluated step sets differ (union={union_desc}) — "
        + "; ".join(bits)
    )


def _enforce_coverage_and_step_parity(results: list[RunResult]) -> None:
    seeds_by_env: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    steps_by_env: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    latest_by_env: dict[str, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))

    for r in results:
        env_id = str(r.env_id)
        method = str(r.method)
        seed = int(r.seed)
        step = int(r.ckpt_step)

        seeds_by_env[env_id][method].add(seed)

        if step >= 0:
            steps_by_env[env_id][method].add(step)
            prev = latest_by_env[env_id][method].get(seed)
            if prev is None or step > prev:
                latest_by_env[env_id][method][seed] = step

    for env_id in sorted(seeds_by_env.keys()):
        methods = seeds_by_env[env_id]
        if len(methods) <= 1:
            continue

        union_seeds: set[int] = set()
        for ss in methods.values():
            union_seeds |= set(ss)

        missing_by_method: dict[str, list[int]] = {}
        for m, ss in methods.items():
            miss = sorted(union_seeds - set(ss))
            if miss:
                missing_by_method[str(m)] = miss

        if missing_by_method:
            raise RuntimeError(_coverage_msg_seed(str(env_id), missing_by_method))

        method_names = sorted((str(m) for m in methods.keys()), key=lambda s: s)

        for seed in sorted(union_seeds):
            latest_by_method: dict[str, int] = {}
            for m in method_names:
                latest_by_method[m] = int(latest_by_env.get(env_id, {}).get(m, {}).get(int(seed), -1))

            vals = [int(v) for v in latest_by_method.values() if int(v) >= 0]
            if not vals:
                continue

            if max(vals) != min(vals):
                raise RuntimeError(_coverage_msg_latest_step(str(env_id), int(seed), latest_by_method))

        union_steps: set[int] = set()
        for m in method_names:
            union_steps |= set(steps_by_env.get(env_id, {}).get(m, set()))

        missing_steps_by_method: dict[str, list[int]] = {}
        for m in method_names:
            have = set(steps_by_env.get(env_id, {}).get(m, set()))
            miss = sorted(int(s) for s in (union_steps - have))
            if miss:
                missing_steps_by_method[m] = miss

        if missing_steps_by_method:
            raise RuntimeError(_coverage_msg_step_sets(str(env_id), union_steps, missing_steps_by_method))


def _write_eval_meta(
    *,
    results_root: Path,
    runs_root: Path,
    episodes: int,
    device: str,
    policy_mode: str,
    interval_steps_values: list[int],
) -> Path:
    meta = {
        "ckpt_policy": "every_k",
        "every_k_source": "cfg.evaluation.interval_steps",
        "interval_steps_values": [int(x) for x in interval_steps_values],
        "episodes": int(episodes),
        "device": str(device),
        "policy_mode": str(policy_mode),
        "strict_coverage": True,
        "strict_step_parity": True,
        "runs_root": str(Path(runs_root).resolve()),
        "results_dir": str(Path(results_root).resolve()),
    }
    out_path = Path(results_root) / "eval_meta.json"
    atomic_write_text(out_path, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return out_path


def run_eval_suite(
    *,
    runs_root: Path,
    results_dir: Path,
) -> None:
    pm = normalize_policy_mode(
        DEFAULT_EVAL_POLICY_MODE, allowed=("mode", "sample"), name="policy_mode"
    )

    root = Path(runs_root).resolve()
    results_root = Path(results_dir).resolve()
    typer.echo(f"[suite] Eval defaults: runs_root={root}, results_dir={results_root}, policy_mode={pm}")

    if not root.exists():
        typer.echo(f"[suite] No runs_root directory found: {root}")
        return

    discovered = discover_run_dirs_with_latest_ckpt(root)
    if not discovered:
        typer.echo(f"[suite] No checkpoints found under {root}.")
        return

    intervals_by_run: dict[Path, int] = {}
    episodes_by_run: dict[Path, int] = {}
    devices_by_run: dict[Path, str] = {}
    selected: list[tuple[Path, Path]] = []

    for rd, ckpt_latest in discovered:
        try:
            payload_latest = load_checkpoint(ckpt_latest, map_location="cpu")
        except Exception as exc:
            typer.echo(f"[suite]    ! {rd.name}: failed to load {ckpt_latest.name} ({exc})")
            continue

        episodes_by_run[rd] = int(_eval_episodes(payload_latest))
        interval = int(_eval_interval_steps(payload_latest))
        intervals_by_run[rd] = int(interval)
        devices_by_run[rd] = str(_eval_device(payload_latest))

        if interval > 0:
            ckpts = select_ckpts_for_run(rd, policy="every_k", every_k=int(interval))
        else:
            ckpts = select_ckpts_for_run(rd, policy="latest")

        for ckpt in ckpts:
            selected.append((rd, Path(ckpt)))

    if not selected:
        typer.echo(f"[suite] No evaluable checkpoints found under {root}.")
        return

    ep_vals = sorted({int(v) for v in episodes_by_run.values() if int(v) > 0})
    if not ep_vals:
        raise RuntimeError("[suite]    ! No positive evaluation episodes found across runs.")
    if len(ep_vals) == 1:
        episodes = int(ep_vals[0])
    else:
        raise RuntimeError(f"[suite]    ! Evaluation episodes mismatch across runs: {ep_vals}")

    dev_vals = sorted(
        {str(devices_by_run.get(rd, "cpu")).strip() for rd, _ in selected if str(rd).strip()}
    )
    dev_vals = [d for d in dev_vals if d]
    if not dev_vals:
        device = "cpu"
    elif len(dev_vals) == 1:
        device = str(dev_vals[0])
    else:
        raise RuntimeError(f"[suite]    ! Evaluation device mismatch across runs: {dev_vals}")

    interval_vals = sorted({int(v) for v in intervals_by_run.values()})
    n_runs = len({rd.resolve() for rd, _ in selected})
    typer.echo(
        f"[suite] Evaluating {len(selected)} checkpoint(s) from {n_runs} run(s) under {root} "
        f"(device={device}, episodes={episodes}, policy_mode={pm}, interval_steps={interval_vals} (0=latest))"
    )

    traj_root = results_root / "plots" / "trajectories"
    traj_root.mkdir(parents=True, exist_ok=True)

    specs: list[EvalCheckpoint] = []

    for rd, ckpt in selected:
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

            notes: list[str] = []

            if cfg_env_id is None:
                notes.append(
                    f"[suite]    ! {rd.name}: cfg.env.id missing in checkpoint; "
                    f"falling back to run dir label {run_env_tag!r}."
                )

            env_for_eval = cfg_env_id or str(run_env_tag or "UnknownEnv")

            if cfg_env_id is not None and run_env_tag is not None:
                expected_tag = str(cfg_env_id).replace("/", "-")
                if str(run_env_tag) != expected_tag:
                    notes.append(
                        f"[suite]    ! {rd.name}: run dir env tag {run_env_tag!r} "
                        f"!= cfg.env.id {cfg_env_id!r}; using cfg.env.id."
                    )

            if cfg_method is not None and run_method is not None:
                if str(cfg_method).strip().lower() != str(run_method).strip().lower():
                    notes.append(
                        f"[suite]    ! {rd.name}: run dir method {run_method!r} "
                        f"!= cfg.method {cfg_method!r}; using cfg.method."
                    )

            if cfg_seed is not None and run_seed is not None and int(cfg_seed) != int(run_seed):
                notes.append(
                    f"[suite]    ! {rd.name}: run dir seed {run_seed} "
                    f"!= cfg.seed {cfg_seed}; using cfg.seed."
                )

            traj_out_dir = traj_root / rd.name / ckpt.stem
            traj_out_dir.mkdir(parents=True, exist_ok=True)

            specs.append(
                EvalCheckpoint(
                    ckpt=ckpt,
                    env=str(env_for_eval),
                    method=str(cfg_method or run_method or "unknown"),
                    seed=int(cfg_seed) if cfg_seed is not None else None,
                    save_traj=True,
                    traj_out_dir=traj_out_dir,
                    payload=payload,
                    notes=tuple(notes),
                    label=f"{rd.name}/{ckpt.stem}",
                )
            )
        except Exception as exc:
            typer.echo(f"[suite]          ! evaluation failed: {exc}")

    if not specs:
        typer.echo("[suite] No checkpoints eligible for evaluation; nothing to do.")
        return

    def _on_start(_i: int, _n: int, spec: EvalCheckpoint) -> None:
        for msg in spec.notes:
            typer.echo(msg)
        name = spec.label or spec.ckpt.name
        env_disp = spec.env if spec.env is not None else "UnknownEnv"
        typer.echo(f"[suite]    - {name}: ckpt={spec.ckpt.name}, env={env_disp}, episodes={episodes}")

    def _on_error(_i: int, _n: int, _spec: EvalCheckpoint, exc: Exception) -> None:
        typer.echo(f"[suite]          ! evaluation failed: {exc}")

    results = evaluate_checkpoints(
        specs,
        episodes=int(episodes),
        device=str(device),
        policy_mode=str(pm),
        evaluate_fn=evaluate,
        on_start=_on_start,
        on_error=_on_error,
        skip_failures=True,
    )

    if not results:
        typer.echo("[suite] No checkpoints evaluated; nothing to write.")
        return

    results_root.mkdir(parents=True, exist_ok=True)
    raw_path = results_root / "summary_raw.csv"
    write_raw_csv(results, raw_path)
    typer.echo(f"[suite] Wrote per-checkpoint results to {raw_path}")

    coverage_rows, _seeds_by_env, _steps_by_env = _coverage_from_results(results)
    coverage_path = results_root / "coverage.csv"
    _write_coverage_csv(coverage_rows, coverage_path)
    typer.echo(f"[suite] Wrote coverage report to {coverage_path}")

    meta_path = _write_eval_meta(
        results_root=results_root,
        runs_root=root,
        episodes=int(episodes),
        device=str(device),
        policy_mode=str(pm),
        interval_steps_values=sorted({int(v) for v in intervals_by_run.values()}),
    )
    typer.echo(f"[suite] Wrote eval metadata to {meta_path}")

    _enforce_coverage_and_step_parity(results)

    summary_path = results_root / "summary.csv"
    agg_rows = aggregate_results(results)
    write_summary_csv(agg_rows, summary_path)
    typer.echo(f"[suite] Wrote aggregated summary to {summary_path}")

    unique_steps = sorted({int(r.ckpt_step) for r in results if int(r.ckpt_step) >= 0})
    if len(unique_steps) > 1:
        by_step_path = results_root / "summary_by_step.csv"
        by_step_rows = aggregate_results_by_step(results)
        write_summary_by_step_csv(by_step_rows, by_step_path)
        typer.echo(f"[suite] Wrote step-grouped summary to {by_step_path}")
