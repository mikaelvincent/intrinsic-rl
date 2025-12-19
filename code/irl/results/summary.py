from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Literal, Sequence

import numpy as np

from irl.utils.io import atomic_write_csv


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
    policy_mode: str = "mode"
    seed_offset: int = 0
    episode_seeds_hash: str = ""


def write_raw_csv(rows: List[RunResult], path: Path) -> None:
    cols = [
        "method",
        "env_id",
        "seed",
        "policy_mode",
        "seed_offset",
        "episode_seeds_hash",
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

    out_rows: list[dict[str, object]] = []
    for r in rows:
        out_rows.append(
            {
                "method": r.method,
                "env_id": r.env_id,
                "seed": int(r.seed),
                "policy_mode": str(r.policy_mode),
                "seed_offset": int(r.seed_offset),
                "episode_seeds_hash": str(r.episode_seeds_hash or ""),
                "ckpt_step": int(r.ckpt_step),
                "episodes": int(r.episodes),
                "mean_return": f"{float(r.mean_return):.6f}",
                "std_return": f"{float(r.std_return):.6f}",
                "min_return": f"{float(r.min_return):.6f}",
                "max_return": f"{float(r.max_return):.6f}",
                "mean_length": f"{float(r.mean_length):.6f}",
                "std_length": f"{float(r.std_length):.6f}",
                "ckpt_path": str(r.ckpt_path),
            }
        )

    atomic_write_csv(path, cols, out_rows)


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


def aggregate_results(
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
                "per_seed_ckpt_policy": str(per_seed_policy),
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


def write_summary_csv(agg_rows: List[Dict[str, object]], path: Path) -> None:
    cols = [
        "method",
        "env_id",
        "per_seed_ckpt_policy",
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
    atomic_write_csv(path, cols, agg_rows)


def aggregate_results_by_step(
    rows: List[RunResult],
    *,
    per_seed_policy: Literal["latest_ckpt"] = "latest_ckpt",
    ci: float = 0.95,
    n_boot: int = 2000,
) -> List[Dict[str, object]]:
    groups: dict[tuple[str, str, int], list[RunResult]] = {}
    for r in rows:
        key = (str(r.method), str(r.env_id), int(r.ckpt_step))
        groups.setdefault(key, []).append(r)

    out: list[dict[str, object]] = []
    for (method, env_id, ckpt_step), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        n_runs_total = int(len(rs))

        rs_per_seed = _reduce_to_one_per_seed(rs, policy=per_seed_policy)
        seeds = sorted({int(r.seed) for r in rs_per_seed})
        n_seeds = int(len(seeds))

        ret_vals = [float(r.mean_return) for r in rs_per_seed]
        len_vals = [float(r.mean_length) for r in rs_per_seed]

        mean_return_mean = float(mean(ret_vals)) if n_seeds > 0 else 0.0
        mean_return_std = float(pstdev(ret_vals)) if n_seeds > 1 else 0.0
        mean_return_se = float(mean_return_std / sqrt(n_seeds)) if n_seeds > 1 else 0.0

        mean_length_mean = float(mean(len_vals)) if n_seeds > 0 else 0.0
        mean_length_std = float(pstdev(len_vals)) if n_seeds > 1 else 0.0
        mean_length_se = float(mean_length_std / sqrt(n_seeds)) if n_seeds > 1 else 0.0

        rng_ret = np.random.default_rng(_stable_u32_seed("ret_step", method, env_id, str(int(ckpt_step))))
        rng_len = np.random.default_rng(_stable_u32_seed("len_step", method, env_id, str(int(ckpt_step))))
        mean_return_ci95_lo, mean_return_ci95_hi = _bootstrap_mean_ci(
            ret_vals, n_boot=int(n_boot), ci=float(ci), rng=rng_ret
        )
        mean_length_ci95_lo, mean_length_ci95_hi = _bootstrap_mean_ci(
            len_vals, n_boot=int(n_boot), ci=float(ci), rng=rng_len
        )

        out.append(
            {
                "method": method,
                "env_id": env_id,
                "ckpt_step": int(ckpt_step),
                "per_seed_ckpt_policy": str(per_seed_policy),
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
            }
        )

    return out


def write_summary_by_step_csv(agg_rows: List[Dict[str, object]], path: Path) -> None:
    cols = [
        "method",
        "env_id",
        "ckpt_step",
        "per_seed_ckpt_policy",
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
    ]
    atomic_write_csv(path, cols, agg_rows)
