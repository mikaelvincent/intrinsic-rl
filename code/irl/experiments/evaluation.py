from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import typer

from irl.evaluator import evaluate
from irl.plot import _parse_run_name
from irl.sweep import RunResult, _aggregate, _find_latest_ckpt, _write_raw_csv, _write_summary_csv
from irl.utils.checkpoint import load_checkpoint


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


def run_eval_suite(
    runs_root: Path, results_dir: Path, episodes: int, device: str, policy_mode: str = "mode"
) -> None:
    pm = str(policy_mode).strip().lower()
    if pm not in {"mode", "sample"}:
        raise typer.BadParameter("--policy must be one of: mode, sample")

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

        try:
            payload = load_checkpoint(ckpt, map_location="cpu")
            step = int(payload.get("step", -1))

            cfg_env_id, cfg_method, cfg_seed = _cfg_fields(payload)

            info = _parse_run_name(rd)
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

            summary = evaluate(
                env=str(env_for_eval),
                ckpt=ckpt,
                episodes=episodes,
                device=device,
                save_traj=True,
                traj_out_dir=traj_out_dir,
                policy_mode=pm,
            )

            results.append(
                RunResult(
                    method=(cfg_method or run_method or "unknown"),
                    env_id=str(summary["env_id"]),
                    seed=int(cfg_seed if cfg_seed is not None else summary.get("seed", run_seed or 0)),
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
