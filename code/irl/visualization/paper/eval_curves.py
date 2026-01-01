from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from irl.visualization.labels import add_legend_rows_top, add_row_label, env_label, legend_ncol, method_label, slugify
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic, sort_env_ids as _sort_env_ids
from irl.visualization.style import DPI, FIG_WIDTH, apply_grid, alpha_for_method, legend_order
from irl.visualization.style import linestyle_for_method, linewidth_for_method, zorder_for_method
from .thresholds import SOLVED_THRESHOLD_LABEL, add_solved_threshold_line, solved_threshold_legend_handle

SCATTER_POINT_SIZE: float = 9.0
SCATTER_OFFSET_P_SCALE: float = 0.25


def _eval_curve_linestyle(method_key: str) -> str | tuple[int, tuple[int, ...]]:
    # Eval curves encode method identity by color; keep GLPE variants solid to match reward-decomp figures.
    k = str(method_key).strip().lower()
    if k.startswith("glpe_"):
        return "-"
    return linestyle_for_method(k)


def _is_ablation_suffix(filename_suffix: str) -> bool:
    return "ablation" in str(filename_suffix).strip().lower()


def _has_glpe_and_variant(method_keys: Sequence[str]) -> bool:
    keys = {str(k).strip().lower() for k in method_keys if str(k).strip()}
    if "glpe" not in keys:
        return False
    return any(k.startswith("glpe_") for k in keys)


def _load_summary_raw(path: Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None

    try:
        df = pd.read_csv(p)
    except Exception:
        return None

    required = {"method", "env_id", "seed", "ckpt_step", "mean_return"}
    if not required.issubset(set(df.columns)):
        return None

    out = df.copy()
    out["env_id"] = out["env_id"].astype(str).str.strip()
    out["method"] = out["method"].astype(str).str.strip()
    out["method_key"] = out["method"].str.lower().str.strip()

    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out["ckpt_step"] = pd.to_numeric(out["ckpt_step"], errors="coerce")
    out["mean_return"] = pd.to_numeric(out["mean_return"], errors="coerce")
    out = out.dropna(subset=["env_id", "method_key", "seed", "ckpt_step", "mean_return"]).copy()

    out["seed"] = out["seed"].astype(int)
    out["ckpt_step"] = out["ckpt_step"].astype(int)

    if "policy_mode" in out.columns:
        out["policy_mode"] = out["policy_mode"].astype(str).str.strip().str.lower()
        if "mode" in set(out["policy_mode"].unique().tolist()):
            out = out.loc[out["policy_mode"] == "mode"].copy()

    return out


def _point_width_in_x_units(ax, fig, *, point_size: float) -> float:
    if str(ax.get_xscale()).strip().lower() != "linear":
        return 0.0

    try:
        x0, x1 = ax.get_xlim()
    except Exception:
        return 0.0

    span = float(x1 - x0)
    if not math.isfinite(span) or span == 0.0:
        return 0.0

    pos = ax.get_position()
    width_px = float(pos.width) * float(fig.get_figwidth()) * float(fig.dpi)
    if not math.isfinite(width_px) or width_px <= 1.0:
        return 0.0

    diam_pt = math.sqrt(max(0.0, float(point_size)))
    if diam_pt <= 0.0:
        return 0.0

    diam_px = diam_pt * float(fig.dpi) / 72.0
    return float(diam_px) * (span / width_px)


def _method_offset_multipliers(methods: Sequence[str]) -> dict[str, float]:
    ms = [str(m).strip().lower() for m in methods if str(m).strip()]
    n = int(len(ms))
    if n <= 0:
        return {}
    center = (float(n) - 1.0) / 2.0
    return {m: float(i) - center for i, m in enumerate(ms)}


def plot_eval_curves_by_env(
    by_step_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
    summary_raw_csv: Path | None = None,
) -> list[Path]:
    _ = title
    if by_step_df is None or by_step_df.empty:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    want = legend_order(methods_to_plot)
    if not want:
        return []

    ablation_mode = _is_ablation_suffix(filename_suffix)

    df = by_step_df.copy()
    df["env_id"] = df["env_id"].astype(str).str.strip()
    if "method_key" not in df.columns:
        if "method" not in df.columns:
            return []
        df["method_key"] = df["method"].astype(str).str.strip().str.lower()
    df["method_key"] = df["method_key"].astype(str).str.strip().str.lower()

    raw_df = _load_summary_raw(Path(summary_raw_csv)) if summary_raw_csv is not None else None

    env_recs: list[tuple[str, pd.DataFrame, list[str]]] = []
    methods_union: set[str] = set()

    for env_id in _sort_env_ids(df["env_id"].unique().tolist()):
        df_env = df.loc[df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        methods_present_set = set(df_env["method_key"].tolist()) & set(want)
        methods_present = [m for m in want if m in methods_present_set]
        if not methods_present:
            continue

        uniq_steps = sorted(set(df_env["ckpt_step"].tolist())) if "ckpt_step" in df_env.columns else []
        if len(uniq_steps) <= 1:
            continue

        if ablation_mode and not _has_glpe_and_variant(methods_present):
            continue

        env_recs.append((str(env_id), df_env, methods_present))
        methods_union |= set(methods_present)

    if not env_recs:
        return []

    legend_methods = legend_order([m for m in want if m in methods_union])

    nrows = int(len(env_recs))
    height = max(2.8, 2.2 * float(nrows))

    plt = apply_rcparams_paper()
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(float(FIG_WIDTH), float(height)),
        dpi=int(DPI),
        squeeze=False,
    )

    has_solved_threshold = False

    for i, (env_id, df_env, methods_present) in enumerate(env_recs):
        ax = axes[i, 0]

        methods_draw = list(methods_present)
        for mk in methods_draw:
            df_m = df_env.loc[df_env["method_key"] == mk].copy()
            if df_m.empty:
                continue
            df_m = df_m.sort_values("ckpt_step").drop_duplicates(subset=["ckpt_step"], keep="last")

            x = df_m["ckpt_step"].to_numpy(dtype=np.int64, copy=False)
            y = pd.to_numeric(df_m["mean_return_mean"], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(x.astype(np.float64)) & np.isfinite(y)
            if not bool(ok.any()):
                continue
            x = x[ok]
            y = y[ok]

            ax.plot(
                x,
                y,
                color=_color_for_method(mk),
                lw=float(linewidth_for_method(mk)),
                ls=_eval_curve_linestyle(mk),
                alpha=float(alpha_for_method(mk)),
                zorder=int(zorder_for_method(mk)),
            )

        thr = add_solved_threshold_line(ax, str(env_id))
        if thr is not None:
            has_solved_threshold = True
        apply_grid(ax)

        ax.set_ylabel("Mean return")
        if i == nrows - 1:
            ax.set_xlabel("Training steps")
        else:
            ax.tick_params(axis="x", which="both", labelbottom=False)

        add_row_label(ax, env_label(env_id))

        if raw_df is not None:
            df_raw_env = raw_df.loc[raw_df["env_id"] == str(env_id)].copy()
            if not df_raw_env.empty:
                methods_offset = list(methods_present)

                offset_order = list(methods_offset)
                if "glpe" in offset_order:
                    tail: list[str] = []
                    for k in ("glpe", "glpe_nogate"):
                        if k in offset_order:
                            offset_order = [m for m in offset_order if m != k]
                            tail.append(k)
                    offset_order = offset_order + tail

                mult = _method_offset_multipliers(offset_order)

                try:
                    x0, x1 = ax.get_xlim()
                    span_x = float(x1 - x0)
                except Exception:
                    span_x = 0.0

                p = _point_width_in_x_units(ax, fig, point_size=float(SCATTER_POINT_SIZE))
                if not math.isfinite(p) or p <= 0.0:
                    p = 0.005 * abs(float(span_x)) if math.isfinite(span_x) else 0.0
                p = float(p) * float(SCATTER_OFFSET_P_SCALE)

                if math.isfinite(p) and p != 0.0 and mult:
                    for mk in methods_offset:
                        df_pts = df_raw_env.loc[df_raw_env["method_key"] == mk]
                        if df_pts.empty:
                            continue

                        x_pts = pd.to_numeric(df_pts["ckpt_step"], errors="coerce").to_numpy(
                            dtype=np.float64, copy=False
                        )
                        y_pts = pd.to_numeric(df_pts["mean_return"], errors="coerce").to_numpy(
                            dtype=np.float64, copy=False
                        )

                        ok = np.isfinite(x_pts) & np.isfinite(y_pts)
                        if not bool(ok.any()):
                            continue

                        x_pts = x_pts[ok]
                        y_pts = y_pts[ok]

                        x_off = x_pts + float(mult.get(mk, 0.0)) * float(p)

                        ax.scatter(
                            x_off,
                            y_pts,
                            s=float(SCATTER_POINT_SIZE),
                            c=_color_for_method(mk),
                            marker="o",
                            edgecolors="none",
                            linewidths=0.0,
                            alpha=0.5,
                            zorder=1.5 if mk == "glpe" else 1.0,
                        )

    handles = []
    labels = []
    for mk in legend_methods:
        handles.append(
            plt.Line2D(
                [],
                [],
                color=_color_for_method(mk),
                lw=float(linewidth_for_method(mk)),
                ls=_eval_curve_linestyle(mk),
                alpha=float(alpha_for_method(mk)),
            )
        )
        labels.append(method_label(mk))

    rows = []
    if handles:
        rows.append((handles, labels, legend_ncol(len(handles))))
    if has_solved_threshold:
        rows.append(([solved_threshold_legend_handle(plt)], [SOLVED_THRESHOLD_LABEL], 1))

    top = 1.0
    if rows:
        top = add_legend_rows_top(fig, rows)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, float(top)])

    out = plots_root / f"eval-curves-{slugify(filename_suffix)}.png"
    save_fig_atomic(fig, out)
    plt.close(fig)
    return [out]
