from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import typer

from .plot_utils import apply_rcparams_paper, save_fig_atomic
from .data import read_scalars


def _style():
    return apply_rcparams_paper()


def _save_fig(fig, path: Path) -> None:
    save_fig_atomic(fig, Path(path))


def _tail_frame(df: pd.DataFrame, *, tail_frac: float, min_rows: int, max_rows: int) -> pd.DataFrame:
    n = int(len(df))
    if n <= 0:
        return df
    frac = float(tail_frac)
    if not (0.0 < frac <= 1.0):
        frac = 1.0
    k = int(np.ceil(frac * n))
    k = int(max(int(min_rows), min(int(max_rows), k)))
    return df.tail(k)


def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.zeros((len(df),), dtype=np.float64), index=df.index)
    s = pd.to_numeric(df[name], errors="coerce").astype(np.float64)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return s.clip(lower=0.0)


def _component_medians(df_tail: pd.DataFrame) -> dict[str, float] | None:
    if df_tail.empty:
        return None

    env_step = _num_col(df_tail, "time_rollout_env_step_s")
    policy = _num_col(df_tail, "time_rollout_policy_s")
    intrinsic_step = _num_col(df_tail, "time_rollout_intrinsic_step_s")
    intrinsic_compute = _num_col(df_tail, "time_intrinsic_compute_s")
    intrinsic_update = _num_col(df_tail, "time_intrinsic_update_s")

    intrinsic = intrinsic_step + intrinsic_compute + intrinsic_update
    gae = _num_col(df_tail, "time_gae_s")
    ppo = _num_col(df_tail, "time_ppo_s")
    other = _num_col(df_tail, "time_rollout_other_s")

    comp = pd.DataFrame(
        {
            "env_step": env_step,
            "policy": policy,
            "intrinsic": intrinsic,
            "gae": gae,
            "ppo": ppo,
            "other": other,
        }
    ).clip(lower=0.0)

    meds = comp.median()
    return {k: float(meds.get(k, 0.0)) for k in comp.columns.tolist()}


def _method_order(methods: Sequence[str]) -> list[str]:
    order = [
        "vanilla",
        "icm",
        "rnd",
        "ride",
        "riac",
        "glpe_lp_only",
        "glpe_impact_only",
        "glpe_nogate",
        "glpe_cache",
        "glpe",
    ]
    idx = {m: i for i, m in enumerate(order)}
    ms = list(methods)

    def key(m: str) -> tuple[int, str]:
        ml = str(m).strip().lower()
        if ml in idx:
            return idx[ml], ml
        if ml.startswith("glpe_"):
            return 90, ml
        return 100, ml

    return sorted(ms, key=key)


def plot_timing_breakdown(
    groups_by_env: Mapping[str, Mapping[str, list[Path]]],
    *,
    plots_root: Path,
    tail_frac: float = 0.25,
    min_tail_rows: int = 8,
    max_tail_rows: int = 50,
) -> list[Path]:
    if not isinstance(groups_by_env, Mapping):
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    plt = _style()

    components = [
        ("env_step", "Env step", "#1f77b4"),
        ("policy", "Policy", "#ff7f0e"),
        ("intrinsic", "Intrinsic", "#d62728"),
        ("gae", "GAE", "#2ca02c"),
        ("ppo", "PPO", "#9467bd"),
        ("other", "Other", "#7f7f7f"),
    ]
    comp_keys = [k for k, _, _ in components]

    written: list[Path] = []

    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: str(kv[0])):
        if not isinstance(by_method, Mapping):
            continue

        method_rows: list[tuple[str, dict[str, float], float, float, int]] = []

        for method, run_dirs in by_method.items():
            if not isinstance(run_dirs, (list, tuple)):
                continue

            per_run: list[dict[str, float]] = []
            totals: list[float] = []

            for rd in run_dirs:
                try:
                    df = read_scalars(Path(rd))
                except Exception:
                    continue
                if df.empty:
                    continue

                tail = _tail_frame(
                    df,
                    tail_frac=float(tail_frac),
                    min_rows=int(min_tail_rows),
                    max_rows=int(max_tail_rows),
                )
                meds = _component_medians(tail)
                if meds is None:
                    continue

                per_run.append(meds)
                totals.append(float(sum(meds.get(k, 0.0) for k in comp_keys)))

            if not per_run:
                continue

            comp_mean: dict[str, float] = {}
            for k in comp_keys:
                comp_mean[k] = float(np.mean([r.get(k, 0.0) for r in per_run]))

            total_mean = float(np.mean(totals))
            total_se = (
                float(np.std(totals, ddof=0) / np.sqrt(float(len(totals))))
                if len(totals) > 1
                else 0.0
            )
            method_rows.append((str(method), comp_mean, total_mean, total_se, int(len(totals))))

        if not method_rows:
            continue

        method_rows.sort(key=lambda x: (str(x[0]).strip().lower(), str(x[0])))
        methods = _method_order([m for m, *_ in method_rows])
        by_name = {m: (cm, tm, te, n) for m, cm, tm, te, n in method_rows}

        x = np.arange(len(methods), dtype=np.float64)
        width = 0.75

        fig, ax = plt.subplots(figsize=(max(7.0, 1.2 + 0.9 * len(methods)), 4.8))

        bottom = np.zeros((len(methods),), dtype=np.float64)
        for key, label, color in components:
            vals = np.asarray([by_name[m][0].get(key, 0.0) for m in methods], dtype=np.float64)
            ax.bar(
                x,
                vals,
                width=width,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.88,
                label=label,
            )
            bottom = bottom + vals

        totals = np.asarray([by_name[m][1] for m in methods], dtype=np.float64)
        ses = np.asarray([by_name[m][2] for m in methods], dtype=np.float64)
        ax.errorbar(
            x,
            totals,
            yerr=ses,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            alpha=0.9,
        )

        for i, m in enumerate(methods):
            n = int(by_name[m][3])
            ax.text(
                float(x[i]),
                float(totals[i]) + 0.02 * float(max(1e-9, totals.max())),
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in methods], rotation=25, ha="right")
        ax.set_ylabel("Seconds per update")
        ax.set_title(f"{env_id} — Per-update runtime breakdown", loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)

        ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.15))

        note = (
            f"Window: last {int(round(100.0 * float(tail_frac)))}% of logged updates per run; "
            "bars=mean across runs; error=SE(total)"
        )
        fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, alpha=0.9)

        env_tag = str(env_id).replace("/", "-")
        out_path = Path(plots_root) / f"{env_tag}__timing_breakdown.png"
        _save_fig(fig, out_path)
        plt.close(fig)

        written.append(out_path)
        typer.echo(f"[suite] Saved timing plot: {out_path}")

    return written
