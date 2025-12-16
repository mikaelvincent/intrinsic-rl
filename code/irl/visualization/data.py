from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


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
    dirs: list[Path] = []
    for pat in patterns:
        for hit in glob.glob(pat):
            p = Path(hit)
            if p.is_file() and p.name == "scalars.csv" and p.parent.name == "logs":
                dirs.append(p.parent.parent)
            elif p.is_dir():
                if (p / "logs" / "scalars.csv").exists():
                    dirs.append(p)
    return _dedup_paths(dirs)


def _parse_run_name(run_dir: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    name = run_dir.name
    parts = name.split("__")
    if len(parts) >= 1:
        info["method"] = parts[0]
    if len(parts) >= 2:
        info["env"] = parts[1]
    if len(parts) >= 3:
        m = re.match(r"seed(\d+)", parts[2])
        if m:
            info["seed"] = m.group(1)
    return info


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


def _aggregate_runs(run_dirs: Sequence[Path], metric: str, smooth: int = 1) -> AggregateResult:
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

        if metric not in df.columns:
            fallback = None
            if metric != "reward_total_mean" and "reward_total_mean" in df.columns:
                fallback = "reward_total_mean"
            elif metric != "reward_mean" and "reward_mean" in df.columns:
                fallback = "reward_mean"

            if fallback is None:
                continue
            metric_local = fallback
        else:
            metric_local = metric

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

    all_steps = sorted(set().union(*[set(s.index) for s in series_per_run]))
    means: list[float] = []
    stds: list[float] = []

    for st in all_steps:
        vals = [float(s[st]) for s in series_per_run if st in s.index]
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0)

    steps_arr = np.asarray(all_steps, dtype=np.int64)
    mean_arr = np.asarray(means, dtype=np.float64)
    std_arr = np.asarray(stds, dtype=np.float64)

    method_hint = list(method_cand)[0] if len(method_cand) == 1 else None
    env_hint = list(env_cand)[0] if len(env_cand) == 1 else None

    return AggregateResult(
        steps=steps_arr,
        mean=mean_arr,
        std=std_arr,
        n_runs=len(series_per_run),
        method_hint=method_hint,
        env_hint=env_hint,
    )
