from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from irl.visualization.labels import add_legend_rows_top, add_row_label, env_label, method_label, slugify
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic, sort_env_ids as _sort_env_ids
from irl.visualization.style import DPI, FIG_WIDTH, apply_grid, legend_order as _legend_order

from .thresholds import SOLVED_THRESHOLD_LABEL, add_solved_threshold_line, solved_threshold_legend_handle


_LABEL_TEXT_PAD_FRAC: float = 0.03
_LABEL_BG_ALPHA: float = 0.25


def _wants_combined_plot(filename_suffix: str) -> bool:
    s = str(filename_suffix).strip().lower()
    return ("baseline" in s) or ("ablation" in s)


def _y_text_for_patch(patch: object, *, use_log: bool, pad_frac: float) -> float:
    try:
        y0 = float(getattr(patch, "get_y")())
        h = float(getattr(patch, "get_height")())
    except Exception:
        return float("nan")

    if not np.isfinite(y0) or not np.isfinite(h):
        return float("nan")

    y1 = y0 + h
    if not np.isfinite(y1):
        return float("nan")

    frac = float(np.clip(float(pad_frac), 0.0, 0.5))

    if not use_log:
        return float(y0 + frac * h)

    if y0 <= 0.0 or y1 <= 0.0:
        return float(y0 + frac * h)

    ratio = float(y1 / y0)
    if not np.isfinite(ratio) or ratio <= 1.0:
        return float(y0)

    return float(y0 * (ratio**frac))


def _annotate_bar_label(ax, patch: object, text: str) -> None:
    try:
        cx = float(getattr(patch, "get_x")()) + 0.5 * float(getattr(patch, "get_width")())
    except Exception:
        return

    try:
        h = float(getattr(patch, "get_height")())
    except Exception:
        h = 0.0

    use_log = str(getattr(ax, "get_yscale", lambda: "linear")()).strip().lower() == "log"
    y = _y_text_for_patch(patch, use_log=bool(use_log), pad_frac=float(_LABEL_TEXT_PAD_FRAC))
    if not (np.isfinite(float(cx)) and np.isfinite(float(y))):
        return

    bbox = {
        "boxstyle": "round,pad=0.12",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": float(_LABEL_BG_ALPHA),
    }

    va = "bottom" if float(h) >= 0.0 else "top"

    ax.text(
        float(cx),
        float(y),
        str(text),
        ha="center",
        va=str(va),
        rotation=0,
        fontsize=8,
        color="black",
        bbox=bbox,
        clip_on=True,
        clip_path=patch,
        zorder=40,
    )


def _is_ablation_suffix(filename_suffix: str) -> bool:
    return "ablation" in str(filename_suffix).strip().lower()


def _has_glpe_and_variant(method_keys: Sequence[str]) -> bool:
    keys = {str(k).strip().lower() for k in method_keys if str(k).strip()}
    if "glpe" not in keys:
        return False
    return any(k.startswith("glpe_") for k in keys)


def plot_eval_bars_by_env(
    summary_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
) -> list[Path]:
    _ = title
    if summary_df is None or summary_df.empty:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    combine = _wants_combined_plot(str(filename_suffix))

    df = summary_df.copy()
    if "env_id" not in df.columns:
        return []
    df["env_id"] = df["env_id"].astype(str).str.strip()

    if "method_key" not in df.columns:
        if "method" not in df.columns:
            return []
        df["method_key"] = df["method"].astype(str).str.strip().str.lower()
    df["method_key"] = df["method_key"].astype(str).str.strip().str.lower()

    if combine:
        filename_suffix = "all-methods"
        methods_to_plot = sorted(
            {
                str(m).strip().lower()
                for m in df["method_key"].unique().tolist()
                if str(m).strip()
            }
        )

    want = _legend_order(methods_to_plot)
    if not want:
        return []

    ablation_mode = _is_ablation_suffix(filename_suffix)

    env_recs: list[tuple[str, dict[str, pd.Series], list[str]]] = []

    for env_id in _sort_env_ids(df["env_id"].unique().tolist()):
        df_env = df.loc[df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        rows_by_method: dict[str, pd.Series] = {}
        for _, r in df_env.iterrows():
            mk = str(r.get("method_key", "")).strip().lower()
            if mk:
                rows_by_method[mk] = r

        methods_present = [m for m in want if m in rows_by_method]
        if not methods_present:
            continue

        if ablation_mode and not _has_glpe_and_variant(methods_present):
            continue

        env_recs.append((str(env_id), rows_by_method, methods_present))

    if not env_recs:
        return []

    nrows = int(len(env_recs))
    height = max(2.6, 2.2 * float(nrows))

    plt = apply_rcparams_paper()
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(float(FIG_WIDTH), float(height)),
        dpi=int(DPI),
        squeeze=False,
    )

    has_solved_threshold = False

    for i, (env_id, rows_by_method, methods_present) in enumerate(env_recs):
        ax = axes[i, 0]

        means = np.asarray(
            [float(rows_by_method[m].get("mean_return_mean", float("nan"))) for m in methods_present],
            dtype=np.float64,
        )
        ci_lo = np.asarray(
            [float(rows_by_method[m].get("mean_return_ci95_lo", float("nan"))) for m in methods_present],
            dtype=np.float64,
        )
        ci_hi = np.asarray(
            [float(rows_by_method[m].get("mean_return_ci95_hi", float("nan"))) for m in methods_present],
            dtype=np.float64,
        )
        n_seeds = np.asarray(
            [float(rows_by_method[m].get("n_seeds", float("nan"))) for m in methods_present],
            dtype=np.float64,
        )

        half = 0.5 * np.abs(ci_hi - ci_lo)
        se = half / 1.96
        n_eff = np.where(np.isfinite(n_seeds) & (n_seeds >= 1.0), n_seeds, 1.0)
        stds = se * np.sqrt(n_eff)

        std_direct = np.asarray(
            [float(rows_by_method[m].get("mean_return_std", float("nan"))) for m in methods_present],
            dtype=np.float64,
        )
        stds = np.where(np.isfinite(std_direct), std_direct, stds)
        stds = np.where(np.isfinite(stds) & (stds >= 0.0), stds, 0.0)

        x = np.arange(len(methods_present), dtype=np.float64)
        for j, mk in enumerate(methods_present):
            ax.bar(
                float(x[j]),
                float(means[j]),
                color=_color_for_method(mk),
                alpha=1.0 if str(mk) == "glpe" else 0.9,
                edgecolor="none",
                linewidth=0.0,
                zorder=10 if str(mk) == "glpe" else 2,
            )

        ax.errorbar(
            x,
            means,
            yerr=stds,
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            capthick=0.9,
            alpha=0.9,
            zorder=20,
        )

        thr = add_solved_threshold_line(ax, str(env_id))
        if thr is not None:
            has_solved_threshold = True

        ax.set_ylabel("Mean return")
        ax.set_xticks(x)

        tick_labels = [method_label(mk) for mk in methods_present]
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")

        if i != nrows - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

        apply_grid(ax)
        add_row_label(ax, env_label(env_id))

    axes[-1, 0].set_xlabel("Method")

    rows = []
    if has_solved_threshold:
        rows.append(([solved_threshold_legend_handle(plt)], [SOLVED_THRESHOLD_LABEL], 1))

    top = 1.0
    if rows:
        top = add_legend_rows_top(fig, rows)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, float(top)])

    out = plots_root / f"eval-bars-{slugify(filename_suffix)}.png"
    save_fig_atomic(fig, out)
    plt.close(fig)
    return [out]
