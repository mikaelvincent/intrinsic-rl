from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

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


def _aggregate(rows: List[RunResult]) -> List[Dict[str, object]]:
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

        out.append(
            {
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
