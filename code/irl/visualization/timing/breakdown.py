from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import typer

from irl.visualization.data import read_scalars
from irl.visualization.labels import add_legend_rows_top, add_row_label, env_label, legend_ncol, method_label
from irl.visualization.palette import color_for_component as _color_for_component
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic, sort_env_ids as _sort_env_ids
from irl.visualization.style import DPI, FIG_WIDTH, apply_grid


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


_LABEL_TEXT_PAD_FRAC: float = 0.03
_LABEL_BG_ALPHA: float = 0.25


def _bar_label_pos(patch: object, *, pad_frac: float) -> tuple[float, float, str] | None:
    try:
        x0 = float(getattr(patch, "get_x")())
        w = float(getattr(patch, "get_width")())
        y0 = float(getattr(patch, "get_y")())
        h = float(getattr(patch, "get_height")())
    except Exception:
        return None

    if not (np.isfinite(x0) and np.isfinite(w) and np.isfinite(y0) and np.isfinite(h)):
        return None

    cx = x0 + 0.5 * w
    frac = float(np.clip(float(pad_frac), 0.0, 0.5))
    cy = y0 + frac * h
    va = "bottom" if h >= 0.0 else "top"
    return float(cx), float(cy), va


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

    plt = apply_rcparams_paper()

    components = [
        ("env_step", "Environment step", _color_for_component("env_step")),
        ("policy", "Policy", _color_for_component("policy")),
        ("intrinsic", "Intrinsic", _color_for_component("intrinsic")),
        ("gae", "GAE", _color_for_component("gae")),
        ("ppo", "PPO", _color_for_component("ppo")),
        ("other", "Other", _color_for_component("other")),
    ]
    comp_keys = [k for k, _, _ in components]

    env_ids = _sort_env_ids(list(groups_by_env.keys()))
    per_env: list[dict[str, object]] = []
    max_methods = 0

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
            method_rows.append(
                (str(method).strip().lower(), comp_mean, total_mean, total_se, int(len(totals)))
            )

        if not method_rows:
            continue

        methods_env = _method_order([m for m, *_ in method_rows])
        max_methods = max(max_methods, int(len(methods_env)))

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

    fig_w = float(FIG_WIDTH)
    fig_h = max(3.4, 2.3 * float(len(per_env)))

    fig, axes = plt.subplots(
        int(len(per_env)),
        1,
        figsize=(fig_w, float(fig_h)),
        dpi=int(DPI),
        sharex=False,
        squeeze=False,
    )

    legend_handles = [plt.Line2D([], [], color=col, lw=8.0) for _k, _lab, col in components]
    legend_labels = [str(lab) for _k, lab, _c in components]

    bbox = {
        "boxstyle": "round,pad=0.12",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": float(_LABEL_BG_ALPHA),
    }

    import matplotlib.patches as mpatches

    for i, rec in enumerate(per_env):
        ax = axes[i, 0]
        env_id = str(rec.get("env_id", ""))
        by_name = rec.get("by_name", {})
        methods_env = rec.get("methods_env", [])

        if not isinstance(by_name, dict) or not isinstance(methods_env, list) or not methods_env:
            ax.axis("off")
            continue

        n_methods = int(len(methods_env))
        x = np.arange(n_methods, dtype=np.float64)
        bottom = np.zeros((n_methods,), dtype=np.float64)

        for key, _lab, color in components:
            vals = np.asarray(
                [float(by_name.get(m, ({}, 0.0, 0.0, 0))[0].get(key, 0.0)) for m in methods_env],
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
                alpha=0.9,
                zorder=2,
            )
            bottom = bottom + vals

        totals = np.asarray(
            [float(by_name.get(m, ({}, 0.0, 0.0, 0))[1]) for m in methods_env],
            dtype=np.float64,
        )
        ses = np.asarray(
            [float(by_name.get(m, ({}, 0.0, 0.0, 0))[2]) for m in methods_env],
            dtype=np.float64,
        )
        ns = np.asarray(
            [int(by_name.get(m, ({}, 0.0, 0.0, 0))[3]) for m in methods_env],
            dtype=np.int64,
        )

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

        width = 0.75
        for xi, yi, n in zip(x.tolist(), totals.tolist(), ns.tolist()):
            if int(n) <= 0 or not np.isfinite(float(yi)):
                continue

            left = float(xi) - 0.5 * float(width)
            clip_patch = mpatches.Rectangle((left, 0.0), float(width), float(yi))
            clip_patch.set_transform(ax.transData)

            pos = _bar_label_pos(clip_patch, pad_frac=float(_LABEL_TEXT_PAD_FRAC))
            if pos is None:
                continue

            cx, cy, va = pos
            ax.text(
                float(cx),
                float(cy),
                f"n={int(n)}",
                ha="center",
                va=str(va),
                rotation=90,
                fontsize=8,
                color="black",
                bbox=bbox,
                clip_on=True,
                clip_path=clip_patch,
                zorder=40,
            )

        ax.set_xlim(-0.5, float(n_methods) - 0.5)
        ax.set_ylabel("Seconds per update")
        ax.set_xticks(x)
        ax.set_xticklabels([method_label(m) for m in methods_env], rotation=25, ha="right")
        if i == len(per_env) - 1:
            ax.set_xlabel("Method")

        apply_grid(ax)
        ax.set_axisbelow(True)
        add_row_label(ax, env_label(env_id))

    top = add_legend_rows_top(fig, [(legend_handles, legend_labels, legend_ncol(len(legend_handles)))])
    fig.tight_layout(rect=[0.0, 0.0, 1.0, float(top)])

    out_path = plots_root / "timing-breakdown.png"
    save_fig_atomic(fig, out_path)
    plt.close(fig)

    typer.echo(f"[suite] Saved timing plot: {out_path}")
    return [out_path]
