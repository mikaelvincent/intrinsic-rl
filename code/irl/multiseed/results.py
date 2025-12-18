from __future__ import annotations

import csv
from pathlib import Path

__all__ = [
    "read_summary_raw",
    "values_for_method",
]


def read_summary_raw(path: Path) -> list[dict]:
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


def values_for_method(
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
