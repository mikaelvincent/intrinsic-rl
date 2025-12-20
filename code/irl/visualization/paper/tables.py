from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from irl.methods.spec import paper_method_groups as _paper_method_groups


def paper_method_groups(methods: Sequence[str]) -> tuple[list[str], list[str]]:
    return _paper_method_groups(methods)


def load_eval_summary_table(summary_csv: Path) -> pd.DataFrame:
    p = Path(summary_csv)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Empty summary table: {p}")

    required = {
        "method",
        "env_id",
        "mean_return_mean",
        "mean_return_ci95_lo",
        "mean_return_ci95_hi",
        "n_seeds",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"summary.csv missing columns {missing}: {p}")

    out = df.copy()
    out["env_id"] = out["env_id"].astype(str).str.strip()
    out["method"] = out["method"].astype(str).str.strip()
    out["method_key"] = out["method"].str.lower().str.strip()
    return out


def load_eval_by_step_table(summary_by_step_csv: Path) -> pd.DataFrame:
    p = Path(summary_by_step_csv)
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Empty by-step summary table: {p}")

    required = {
        "method",
        "env_id",
        "ckpt_step",
        "mean_return_mean",
        "mean_return_ci95_lo",
        "mean_return_ci95_hi",
        "n_seeds",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"summary_by_step.csv missing columns {missing}: {p}")

    out = df.copy()
    out["env_id"] = out["env_id"].astype(str).str.strip()
    out["method"] = out["method"].astype(str).str.strip()
    out["method_key"] = out["method"].str.lower().str.strip()
    out["ckpt_step"] = pd.to_numeric(out["ckpt_step"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["ckpt_step"]).copy()
    out["ckpt_step"] = out["ckpt_step"].astype(int)
    return out
