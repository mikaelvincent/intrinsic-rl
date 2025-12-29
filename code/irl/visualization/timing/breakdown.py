from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import typer

from irl.visualization.data import read_scalars
from irl.visualization.palette import color_for_component as _color_for_component
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, LEGEND_FRAMEALPHA, LEGEND_FONTSIZE, apply_grid


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
        "glpe",
        "glpe_lp_only",
        "glpe_impact_only",
        "glpe_nogate",
        "glpe_cache",
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


def _cleanup_timing_breakdown_outputs(plots_root: Path) -> None:
    root = Path(plots_root)
    if not root.exists():
        return
    for p in sorted(root.glob("*__timing_breakdown.png"), key=lambda x: str(x)):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass


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

    _cleanup_timing_breakdown_outputs(plots_root)

    plt = apply_rcparams_paper()

    components = [
        ("env_step", "Env step", _color_for_component("env_step")),
        ("policy", "Policy", _color_for_component("policy")),
        ("intrinsic", "Intrinsic", _color_for_component("intrinsic")),
        ("gae", "GAE", _color_for_component("gae")),
        ("ppo", "PPO", _color_for_component("ppo")),
        ("other", "Other", _color_for_component("other")),
    ]
    comp_keys = [k for k, _, _ in components]

    env_ids = [str(k) for k in sorted(groups_by_env.keys(), key=lambda x: str(x))]
    per_env: list[dict[str, object]] = []
    union_methods: set[str] = set()

    for env_id in env_ids:
        by_method = groups_by_env.get(env_id)
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
            method_rows.append((str(method).strip().lower(), comp_mean, total_mean, total_se, int(len(totals))))

        if not method_rows:
            continue

        methods_env = _method_order([m for m, *_ in method_rows])
        union_methods |= set(methods_env)

        by_name = {m: (cm, tm, te, n) for m, cm, tm, te, n in method_rows}
        per_env.append(
            {
                "env_id": str(env_id),
                "by_name": by_name,
                "methods_env": methods_env,
            }
        )

    if not per_env:
        return []

    methods_all = _method_order(sorted(union_methods))
    max_methods = int(len(methods_all))

    fig_w = max(9.0, 1.2 + 0.9 * float(max_methods))
    fig_h = max(4.8, 3.6 * float(len(per_env)))

    fig, axes = plt.subplots(
        int(len(per_env)),
        1,
        figsize=(float(fig_w), float(fig_h)),
        dpi=int(DPI),
        sharex=True,
        squeeze=False,
    )

    handles = []
    labels = []
    for _k, lab, col in components:
        handles.append(plt.Line2D([], [], color=col, lw=8.0))
        labels.append(str(lab))

    for i, rec in enumerate(per_env):
        ax = axes[i, 0]
        env_id = str(rec.get("env_id", ""))
        by_name = rec.get("by_name", {})
        if not isinstance(by_name, dict):
            by_name = {}

        x = np.arange(len(methods_all), dtype=np.float64)
        bottom = np.zeros((len(methods_all),), dtype=np.float64)

        for key, _lab, color in components:
            vals = np.asarray(
                [float(by_name.get(m, ({}, 0.0, 0.0, 0))[0].get(key, 0.0)) for m in methods_all],
                dtype=np.float64,
            )
            ax.bar(
                x,
                vals,
                width=0.75,
                bottom=bottom,
                color=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.88,
                zorder=2,
            )
            bottom = bottom + vals

        totals = np.asarray([float(by_name.get(m, ({}, 0.0, 0.0, 0))[1]) for m in methods_all], dtype=np.float64)
        ses = np.asarray([float(by_name.get(m, ({}, 0.0, 0.0, 0))[2]) for m in methods_all], dtype=np.float64)
        ns = np.asarray([int(by_name.get(m, ({}, 0.0, 0.0, 0))[3]) for m in methods_all], dtype=np.int64)

        ax.errorbar(
            x,
            totals,
            yerr=ses,
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            capthick=0.9,
            alpha=0.9,
            zorder=10,
        )

        ymax = float(np.nanmax(totals)) if totals.size else 0.0
        yoff = 0.02 * float(max(1e-9, ymax))

        for xi, yi, n in zip(x.tolist(), totals.tolist(), ns.tolist()):
            if int(n) <= 0 or not np.isfinite(float(yi)):
                continue
            ax.text(
                float(xi),
                float(yi) + yoff,
                f"n={int(n)}",
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.9,
                zorder=20,
            )

        ax.set_ylabel("Seconds/update")
        ax.set_title(f"{env_id} — Per-update runtime breakdown")
        apply_grid(ax)
        ax.set_axisbelow(True)

        if i < int(len(per_env)) - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    axes[-1, 0].set_xticks(np.arange(len(methods_all), dtype=np.float64))
    axes[-1, 0].set_xticklabels([str(m) for m in methods_all], rotation=25, ha="right")
    axes[-1, 0].set_xlabel("Method")

    fig.legend(
        handles,
        labels,
        ncol=3,
        framealpha=float(LEGEND_FRAMEALPHA),
        fontsize=int(LEGEND_FONTSIZE),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])

    out_path = plots_root / "suite__timing_breakdown.png"
    save_fig_atomic(fig, out_path)
    plt.close(fig)

    typer.echo(f"[suite] Saved timing plot: {out_path}")
    return [out_path]
