from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from irl.trainer.metrics_schema import (
    GLPE_GATE_ANY_COLS,
    GLPE_REQUIRED_COMPONENT_COLS,
    SCALARS_REQUIRED_COMMON_COLS,
)
from irl.utils.checkpoint import load_checkpoint
from irl.utils.runs import parse_run_name as _parse_run_name


def _csv_header(path: Path) -> list[str] | None:
    try:
        with Path(path).open("r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            row = next(r, None)
    except Exception:
        return None

    if not row:
        return None
    return [str(c).strip() for c in row if str(c).strip()]


def _iter_scalars_csv(runs_root: Path) -> list[Path]:
    root = Path(runs_root).resolve()
    if not root.exists():
        return []
    return sorted(root.rglob("logs/scalars.csv"), key=lambda p: str(p))


def _run_name_from_scalars_path(scalars_csv: Path) -> str:
    try:
        return scalars_csv.parent.parent.name
    except Exception:
        return scalars_csv.parent.name


def _method_from_run_name(run_name: str) -> str:
    info = _parse_run_name(str(run_name))
    method = info.get("method")
    m = (str(method) if method is not None else "").strip().lower()
    return m or "unknown"


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str] | None]:
    try:
        with Path(path).open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if r.fieldnames is None:
                return [], None
            rows = [row for row in r]
            return rows, list(r.fieldnames)
    except Exception:
        return [], None


def _as_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _validate_trajectory_provenance(
    runs_root: Path, results_dir: Path
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    traj_root = Path(results_dir) / "plots" / "trajectories"
    if not traj_root.exists():
        warnings.append(f"Missing trajectories dir: {traj_root}")
        return errors, warnings

    npz_files = sorted(traj_root.rglob("*_trajectory.npz"), key=lambda p: str(p))
    if not npz_files:
        warnings.append(f"No trajectory npz files under: {traj_root}")
        return errors, warnings

    runs_root = Path(runs_root).resolve()

    for p in npz_files:
        try:
            rel = p.resolve().relative_to(traj_root.resolve())
        except Exception:
            rel = Path(p.name)

        if len(rel.parts) < 2:
            errors.append(f"Trajectory missing run provenance in path: {p}")
            continue

        run_name = rel.parts[0]
        if not (runs_root / run_name).exists():
            errors.append(f"Trajectory run dir not found under runs_root: {run_name}")

        try:
            data = np.load(p, allow_pickle=False)
        except Exception as exc:
            errors.append(f"Unreadable trajectory npz: {p} ({type(exc).__name__})")
            continue

        keys = set(map(str, getattr(data, "files", [])))
        for k in ("env_id", "method", "gates", "obs"):
            if k not in keys:
                errors.append(f"Trajectory missing key {k!r}: {p}")

        try:
            env_id = data["env_id"].reshape(-1)[0]
            method = data["method"].reshape(-1)[0]
            if str(env_id).strip() == "" or str(method).strip() == "":
                errors.append(f"Trajectory metadata empty (env_id/method): {p}")
        except Exception:
            errors.append(f"Trajectory metadata unreadable (env_id/method): {p}")

    return errors, warnings


def _validate_plot_metrics(runs_root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    scalars_paths = _iter_scalars_csv(runs_root)
    if not scalars_paths:
        errors.append(f"No logs/scalars.csv found under runs_root: {Path(runs_root).resolve()}")
        return errors, warnings

    required_common = set(SCALARS_REQUIRED_COMMON_COLS)
    required_glpe_components = set(GLPE_REQUIRED_COMPONENT_COLS)
    glpe_gate_any = set(GLPE_GATE_ANY_COLS)

    for sp in scalars_paths:
        run_name = _run_name_from_scalars_path(sp)
        method = _method_from_run_name(run_name)

        hdr = _csv_header(sp)
        if hdr is None:
            errors.append(f"Empty/unreadable scalars.csv header: {sp}")
            continue

        cols = set(hdr)
        missing_common = sorted(required_common - cols)
        if missing_common:
            errors.append(f"Missing scalar columns in {sp}: {missing_common}")

        if method.startswith("glpe"):
            if not (glpe_gate_any & cols):
                errors.append(
                    f"Missing GLPE gating metric in {sp}: require one of {sorted(glpe_gate_any)}"
                )
            missing_comp = sorted(required_glpe_components - cols)
            if missing_comp:
                errors.append(f"Missing GLPE component metrics in {sp}: {missing_comp}")

    return errors, warnings


def _validate_eval_env_ids(results_dir: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    errors: list[str] = []
    warnings: list[str] = []

    raw_path = Path(results_dir) / "summary_raw.csv"
    rows, cols = _read_csv_rows(raw_path)
    if cols is None:
        errors.append(f"Missing or unreadable summary_raw.csv: {raw_path}")
        return errors, warnings, []

    required = {"env_id", "method", "seed", "ckpt_path"}
    missing = sorted(required - set(cols))
    if missing:
        errors.append(f"summary_raw.csv missing columns {missing}: {raw_path}")
        return errors, warnings, rows

    for r in rows:
        env_id = (r.get("env_id") or "").strip()
        ckpt_path_s = (r.get("ckpt_path") or "").strip()
        if not ckpt_path_s:
            errors.append(f"summary_raw row missing ckpt_path for env_id={env_id!r}")
            continue

        ckpt_path = Path(ckpt_path_s)
        if not ckpt_path.exists():
            errors.append(f"Checkpoint missing on disk: {ckpt_path}")
            continue

        try:
            payload = load_checkpoint(ckpt_path, map_location="cpu")
        except Exception as exc:
            errors.append(f"Failed to load checkpoint {ckpt_path} ({type(exc).__name__}: {exc})")
            continue

        cfg = payload.get("cfg") or {}
        cfg_env = None
        try:
            env_cfg = cfg.get("env") or {}
            if isinstance(env_cfg, dict):
                cfg_env = env_cfg.get("id")
        except Exception:
            cfg_env = None

        if cfg_env is None:
            warnings.append(f"Checkpoint cfg.env.id missing: {ckpt_path}")
            continue

        if str(cfg_env) != env_id:
            errors.append(
                f"Eval env_id mismatch for {ckpt_path}: summary_raw env_id={env_id!r} cfg.env.id={str(cfg_env)!r}"
            )

    return errors, warnings, rows


def _validate_seed_coverage_parity(raw_rows: Iterable[dict[str, str]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    seeds_by_env_method: dict[tuple[str, str], set[int]] = {}
    methods_by_env: dict[str, set[str]] = {}

    for r in raw_rows:
        env_id = (r.get("env_id") or "").strip()
        method = (r.get("method") or "").strip()
        seed_i = _as_int(r.get("seed"))
        if not env_id or not method or seed_i is None:
            continue
        k = (env_id, method)
        seeds_by_env_method.setdefault(k, set()).add(int(seed_i))
        methods_by_env.setdefault(env_id, set()).add(method)

    for env_id, methods in sorted(methods_by_env.items(), key=lambda kv: kv[0]):
        if len(methods) <= 1:
            continue

        union: set[int] = set()
        for m in methods:
            union |= seeds_by_env_method.get((env_id, m), set())

        for m in sorted(methods):
            have = seeds_by_env_method.get((env_id, m), set())
            missing = sorted(union - have)
            if missing:
                errors.append(f"Seed coverage mismatch env={env_id} method={m}: missing {missing}")

    if not seeds_by_env_method:
        warnings.append("No (env_id, method, seed) rows found in summary_raw.csv")

    return errors, warnings


def _validate_summary_tables(
    results_dir: Path, raw_rows: Iterable[dict[str, str]]
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    summary_path = Path(results_dir) / "summary.csv"
    rows, cols = _read_csv_rows(summary_path)
    if cols is None:
        errors.append(f"Missing or unreadable summary.csv: {summary_path}")
        return errors, warnings

    required_cols = {
        "method",
        "env_id",
        "n_runs",
        "n_seeds",
        "seeds",
        "mean_return_mean",
        "mean_return_std",
        "mean_return_se",
        "mean_return_ci95_lo",
        "mean_return_ci95_hi",
        "mean_length_mean",
        "mean_length_std",
        "mean_length_se",
        "mean_length_ci95_lo",
        "mean_length_ci95_hi",
        "step_min",
        "step_max",
        "step_mean",
    }
    missing = sorted(required_cols - set(cols))
    if missing:
        errors.append(f"summary.csv missing columns {missing}: {summary_path}")
        return errors, warnings

    raw_seeds: dict[tuple[str, str], set[int]] = {}
    for r in raw_rows:
        env_id = (r.get("env_id") or "").strip()
        method = (r.get("method") or "").strip()
        seed_i = _as_int(r.get("seed"))
        if not env_id or not method or seed_i is None:
            continue
        raw_seeds.setdefault((env_id, method), set()).add(int(seed_i))

    for row in rows:
        env_id = (row.get("env_id") or "").strip()
        method = (row.get("method") or "").strip()
        n_seeds = _as_int(row.get("n_seeds"))
        seeds_s = (row.get("seeds") or "").strip()

        if not env_id or not method:
            errors.append("summary.csv row missing env_id/method")
            continue
        if n_seeds is None or n_seeds < 0:
            errors.append(f"summary.csv invalid n_seeds for env={env_id} method={method}")
            continue

        seeds_list = [s for s in (x.strip() for x in seeds_s.split(",")) if s]
        try:
            seeds_int = [int(s) for s in seeds_list]
        except Exception:
            errors.append(
                f"summary.csv seeds not parseable for env={env_id} method={method}: {seeds_s!r}"
            )
            continue

        uniq = sorted(set(seeds_int))
        if len(uniq) != len(seeds_int):
            errors.append(
                f"summary.csv has duplicate seeds for env={env_id} method={method}: {seeds_s!r}"
            )
        if len(uniq) != int(n_seeds):
            errors.append(
                f"summary.csv n_seeds mismatch env={env_id} method={method}: n_seeds={n_seeds} seeds={uniq}"
            )

        for k in (
            "mean_return_se",
            "mean_return_ci95_lo",
            "mean_return_ci95_hi",
            "mean_length_se",
            "mean_length_ci95_lo",
            "mean_length_ci95_hi",
        ):
            v = _as_float(row.get(k))
            if v is None or not np.isfinite(v):
                errors.append(
                    f"summary.csv invalid {k} env={env_id} method={method}: {row.get(k)!r}"
                )

        lo = _as_float(row.get("mean_return_ci95_lo"))
        hi = _as_float(row.get("mean_return_ci95_hi"))
        if lo is not None and hi is not None and lo > hi:
            errors.append(
                f"summary.csv mean_return CI inverted env={env_id} method={method}: {lo} > {hi}"
            )

        expected = raw_seeds.get((env_id, method))
        if expected is not None and set(uniq) != set(expected):
            errors.append(
                f"summary.csv seeds differ from summary_raw env={env_id} method={method}: "
                f"summary={uniq} raw={sorted(expected)}"
            )

    return errors, warnings


def run_validate_results(*, runs_root: Path, results_dir: Path, strict: bool = True) -> bool:
    import typer

    runs_root_r = Path(runs_root).resolve()
    results_dir_r = Path(results_dir).resolve()

    errors: list[str] = []
    warnings: list[str] = []

    e0, w0 = _validate_plot_metrics(runs_root_r)
    errors.extend(e0)
    warnings.extend(w0)

    e1, w1, raw_rows = _validate_eval_env_ids(results_dir_r)
    errors.extend(e1)
    warnings.extend(w1)

    e2, w2 = _validate_seed_coverage_parity(raw_rows)
    errors.extend(e2)
    warnings.extend(w2)

    e3, w3 = _validate_summary_tables(results_dir_r, raw_rows)
    errors.extend(e3)
    warnings.extend(w3)

    e4, w4 = _validate_trajectory_provenance(runs_root_r, results_dir_r)
    errors.extend(e4)
    warnings.extend(w4)

    for msg in warnings:
        typer.echo(f"[validate] WARN  {msg}")
    for msg in errors:
        typer.echo(f"[validate] ERROR {msg}")

    typer.echo(f"[validate] Done: {len(errors)} error(s), {len(warnings)} warning(s).")

    if errors:
        return False
    if strict and warnings:
        return False
    return True
