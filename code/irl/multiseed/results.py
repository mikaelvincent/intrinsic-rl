from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Literal, Sequence

import numpy as np

from irl.utils.checkpoint import atomic_replace


@dataclass
class RunResult:
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


def _write_raw_csv(rows: List[RunResult], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
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
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except Exception:
            pass
    atomic_replace(tmp, path)


def _stable_u32_seed(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _bootstrap_mean_ci(
    vals: Sequence[float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator,
) -> tuple[float, float]:
    x = np.asarray(list(vals), dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        v = float(x[0])
        return v, v

    B = int(max(1, n_boot))
    idx = rng.integers(0, x.size, size=(B, x.size))
    samples = x[idx].mean(axis=1)

    alpha = (1.0 - float(ci)) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return lo, hi


def _reduce_to_one_per_seed(
    rows: Sequence[RunResult],
    *,
    policy: Literal["latest_ckpt"] = "latest_ckpt",
) -> list[RunResult]:
    if policy != "latest_ckpt":
        raise ValueError(f"Unsupported per-seed reduce policy: {policy!r}")

    by_seed: dict[int, RunResult] = {}
    for r in rows:
        sid = int(r.seed)
        prev = by_seed.get(sid)
        if prev is None:
            by_seed[sid] = r
            continue

        s_new = int(r.ckpt_step)
        s_prev = int(prev.ckpt_step)

        if s_new > s_prev:
            by_seed[sid] = r
            continue

        if s_new == s_prev and str(r.ckpt_path) > str(prev.ckpt_path):
            by_seed[sid] = r

    return [by_seed[s] for s in sorted(by_seed.keys())]


def _step_stats(steps: Sequence[int]) -> tuple[int, int, int]:
    ok = [int(s) for s in steps if int(s) >= 0]
    if not ok:
        return -1, -1, -1
    return int(min(ok)), int(max(ok)), int(round(mean(ok)))


def _aggregate(
    rows: List[RunResult],
    *,
    per_seed_policy: Literal["latest_ckpt"] = "latest_ckpt",
    ci: float = 0.95,
    n_boot: int = 2000,
) -> List[Dict[str, object]]:
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for r in rows:
        key = (r.method, r.env_id)
        groups.setdefault(key, []).append(r)

    out: list[dict[str, object]] = []
    for (method, env_id), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n_runs_total = int(len(rs))

        rs_per_seed = _reduce_to_one_per_seed(rs, policy=per_seed_policy)
        seeds = sorted({int(r.seed) for r in rs_per_seed})
        n_seeds = int(len(seeds))

        ret_vals = [float(r.mean_return) for r in rs_per_seed]
        len_vals = [float(r.mean_length) for r in rs_per_seed]
        step_vals = [int(r.ckpt_step) for r in rs_per_seed]

        mean_return_mean = float(mean(ret_vals)) if n_seeds > 0 else 0.0
        mean_return_std = float(pstdev(ret_vals)) if n_seeds > 1 else 0.0
        mean_return_se = float(mean_return_std / sqrt(n_seeds)) if n_seeds > 1 else 0.0

        mean_length_mean = float(mean(len_vals)) if n_seeds > 0 else 0.0
        mean_length_std = float(pstdev(len_vals)) if n_seeds > 1 else 0.0
        mean_length_se = float(mean_length_std / sqrt(n_seeds)) if n_seeds > 1 else 0.0

        rng_ret = np.random.default_rng(_stable_u32_seed("ret", method, env_id))
        rng_len = np.random.default_rng(_stable_u32_seed("len", method, env_id))
        mean_return_ci95_lo, mean_return_ci95_hi = _bootstrap_mean_ci(
            ret_vals, n_boot=int(n_boot), ci=float(ci), rng=rng_ret
        )
        mean_length_ci95_lo, mean_length_ci95_hi = _bootstrap_mean_ci(
            len_vals, n_boot=int(n_boot), ci=float(ci), rng=rng_len
        )

        step_min, step_max, step_mean = _step_stats(step_vals)

        out.append(
            {
                "method": method,
                "env_id": env_id,
                "episodes_per_seed": int(rs_per_seed[0].episodes) if rs_per_seed else 0,
                "n_runs": n_runs_total,
                "n_seeds": n_seeds,
                "seeds": ",".join(str(s) for s in seeds),
                "mean_return_mean": mean_return_mean,
                "mean_return_std": mean_return_std,
                "mean_return_se": mean_return_se,
                "mean_return_ci95_lo": float(mean_return_ci95_lo),
                "mean_return_ci95_hi": float(mean_return_ci95_hi),
                "mean_length_mean": mean_length_mean,
                "mean_length_std": mean_length_std,
                "mean_length_se": mean_length_se,
                "mean_length_ci95_lo": float(mean_length_ci95_lo),
                "mean_length_ci95_hi": float(mean_length_ci95_hi),
                "step_min": int(step_min),
                "step_max": int(step_max),
                "step_mean": int(step_mean),
            }
        )
    return out


def _write_summary_csv(agg_rows: List[Dict[str, object]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    cols = [
        "method",
        "env_id",
        "episodes_per_seed",
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
    ]
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in agg_rows:
            w.writerow(row)
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except Exception:
            pass
    atomic_replace(tmp, path)


def _read_summary_raw(path: Path) -> list[dict]:
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
    filt = [r for r in raw if r["env_id"] == env and r["method"] == method]
    if not filt:
        return []

    if latest_per_seed:
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
