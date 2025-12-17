from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from irl.utils.runs import expand_runs_from_patterns, parse_run_name

_REWARD_METRIC_FALLBACKS: dict[str, tuple[str, ...]] = {
    "reward_mean": ("reward_total_mean",),
    "reward_total_mean": ("reward_mean",),
}


def _dedup_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            out.append(rp)
            seen.add(rp)
    return out


def _expand_run_dirs(patterns: Sequence[str]) -> list[Path]:
    return _dedup_paths(expand_runs_from_patterns(list(patterns)))


def _parse_run_name(run_dir: Path) -> dict[str, str]:
    return parse_run_name(run_dir.name)


def _read_scalars(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "logs" / "scalars.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing scalars.csv in {run_dir}")
    df = pd.read_csv(path)
    if "step" not in df.columns:
        raise ValueError(f"'step' column not found in {path}")

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)

    if df["step"].duplicated().any():
        df = df.drop_duplicates(subset=["step"], keep="last")

    df = df.sort_values("step").reset_index(drop=True)
    return df


def _smooth_series(s: pd.Series, window: int) -> pd.Series:
    w = int(max(1, window))
    if w == 1:
        return s
    return s.rolling(window=w, min_periods=1).mean()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class AggregateResult:
    steps: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_runs: int
    method_hint: Optional[str]
    env_hint: Optional[str]


def _resolve_metric_for_run(
    *,
    metric: str,
    df_columns: Sequence[str],
    scalars_path: Path,
) -> str | None:
    cols = set(map(str, df_columns))
    if metric in cols:
        return metric

    fallback = None
    for cand in _REWARD_METRIC_FALLBACKS.get(metric, ()):
        if cand in cols:
            fallback = cand
            break

    if fallback is not None:
        warnings.warn(
            f"Metric {metric!r} missing in {scalars_path}; using {fallback!r}. "
            f"Available columns: {sorted(cols)}",
            UserWarning,
        )
        return fallback

    warnings.warn(
        f"Metric {metric!r} missing in {scalars_path}; skipping run. "
        f"Available columns: {sorted(cols)}",
        UserWarning,
    )
    return None


def _aggregate_runs(
    run_dirs: Sequence[Path],
    metric: str,
    smooth: int = 1,
    *,
    align: str = "interpolate",
) -> AggregateResult:
    if not run_dirs:
        return AggregateResult(np.array([]), np.array([]), np.array([]), 0, None, None)

    method_cand: set[str] = set()
    env_cand: set[str] = set()

    series_per_run: list[pd.Series] = []
    for rd in run_dirs:
        info = _parse_run_name(rd)
        if "method" in info:
            method_cand.add(info["method"])
        if "env" in info:
            env_cand.add(info["env"])

        try:
            df = _read_scalars(rd)
        except Exception:
            continue

        scalars_path = rd / "logs" / "scalars.csv"
        metric_local = _resolve_metric_for_run(
            metric=str(metric),
            df_columns=df.columns,
            scalars_path=scalars_path,
        )
        if metric_local is None:
            continue

        y = pd.to_numeric(df[metric_local], errors="coerce")
        x = df["step"]

        s = pd.Series(y.values, index=x.values).dropna()
        if s.index.has_duplicates:
            s = s[~s.index.duplicated(keep="last")]
        s = s.sort_index()
        s = _smooth_series(s, smooth)
        series_per_run.append(s)

    if not series_per_run:
        return AggregateResult(np.array([]), np.array([]), np.array([]), 0, None, None)

    method_hint = list(method_cand)[0] if len(method_cand) == 1 else None
    env_hint = list(env_cand)[0] if len(env_cand) == 1 else None

    mode = str(align).strip().lower()
    if mode not in {"union", "intersection", "interpolate"}:
        raise ValueError("align must be one of: union, intersection, interpolate")

    if mode == "union":
        all_steps = sorted(set().union(*[set(s.index) for s in series_per_run]))
        means = np.empty((len(all_steps),), dtype=np.float64)
        stds = np.empty((len(all_steps),), dtype=np.float64)
        for i, st in enumerate(all_steps):
            vals = [float(s.loc[st]) for s in series_per_run if st in s.index]
            means[i] = float(np.mean(vals)) if vals else float("nan")
            stds[i] = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
        return AggregateResult(
            steps=np.asarray(all_steps, dtype=np.int64),
            mean=means,
            std=stds,
            n_runs=len(series_per_run),
            method_hint=method_hint,
            env_hint=env_hint,
        )

    if mode == "intersection":
        common = set(series_per_run[0].index)
        for s in series_per_run[1:]:
            common &= set(s.index)
        if not common:
            return AggregateResult(np.array([]), np.array([]), np.array([]), 0, method_hint, env_hint)

        steps = np.asarray(sorted(common), dtype=np.int64)
        vals = np.empty((len(series_per_run), steps.size), dtype=np.float64)
        for r_i, s in enumerate(series_per_run):
            vals[r_i, :] = np.asarray([float(s.loc[int(st)]) for st in steps], dtype=np.float64)

        return AggregateResult(
            steps=steps,
            mean=vals.mean(axis=0),
            std=vals.std(axis=0, ddof=0),
            n_runs=len(series_per_run),
            method_hint=method_hint,
            env_hint=env_hint,
        )

    start = max(int(s.index.min()) for s in series_per_run)
    end = min(int(s.index.max()) for s in series_per_run)
    if start > end:
        return AggregateResult(np.array([]), np.array([]), np.array([]), 0, method_hint, env_hint)

    grid_set: set[int] = set()
    for s in series_per_run:
        grid_set |= set(int(v) for v in s.index.values.tolist())
    grid_set.add(int(start))
    grid_set.add(int(end))

    steps = np.asarray(sorted(v for v in grid_set if int(start) <= int(v) <= int(end)), dtype=np.int64)
    if steps.size == 0:
        return AggregateResult(np.array([]), np.array([]), np.array([]), 0, method_hint, env_hint)

    xq = steps.astype(np.float64, copy=False)
    vals = np.empty((len(series_per_run), steps.size), dtype=np.float64)
    for r_i, s in enumerate(series_per_run):
        xs = s.index.to_numpy(dtype=np.float64, copy=False)
        ys = s.to_numpy(dtype=np.float64, copy=False)
        vals[r_i, :] = np.interp(xq, xs, ys)

    return AggregateResult(
        steps=steps,
        mean=vals.mean(axis=0),
        std=vals.std(axis=0, ddof=0),
        n_runs=len(series_per_run),
        method_hint=method_hint,
        env_hint=env_hint,
    )
