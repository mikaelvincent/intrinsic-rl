from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import DPI, FIGSIZE, apply_grid
from .thresholds import add_solved_threshold_line


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


def _finite_minmax(vals: Iterable[float]) -> tuple[float, float] | None:
    arr = np.asarray([float(v) for v in vals], dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.min()), float(arr.max())


def _set_y_minmax(ax, lo: float, hi: float) -> None:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return
    if float(lo) == float(hi):
        pad = 1.0 if abs(float(lo)) < 1.0 else 0.05 * abs(float(lo))
        ax.set_ylim(float(lo) - pad, float(hi) + pad)
        return
    span = float(hi - lo)
    pad = max(1e-6, 0.08 * span)
    ax.set_ylim(float(lo) - pad, float(hi) + pad)


def plot_eval_bars_by_env(
    summary_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
) -> list[Path]:
    if summary_df is None or summary_df.empty:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    want = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
    if not want:
        return []

    label_by_key: dict[str, str] = {}
    if "method_key" in summary_df.columns and "method" in summary_df.columns:
        label_by_key = (
            summary_df.drop_duplicates(subset=["method_key"], keep="first")
            .set_index("method_key")["method"]
            .astype(str)
            .to_dict()
        )
    elif "method" in summary_df.columns:
        tmp = summary_df.copy()
        tmp["method_key"] = tmp["method"].astype(str).str.strip().str.lower()
        label_by_key = (
            tmp.drop_duplicates(subset=["method_key"], keep="first")
            .set_index("method_key")["method"]
            .astype(str)
            .to_dict()
        )

    df = summary_df.copy()
    df["env_id"] = df["env_id"].astype(str).str.strip()
    if "method_key" not in df.columns:
        df["method_key"] = df["method"].astype(str).str.strip().str.lower()
    df["method_key"] = df["method_key"].astype(str).str.strip().str.lower()

    plt = apply_rcparams_paper()
    written: list[Path] = []

    for env_id in sorted(df["env_id"].unique().tolist()):
        df_env = df.loc[df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        rows_by_method: dict[str, Mapping[str, Any]] = {}
        for _, r in df_env.iterrows():
            mk = str(r.get("method_key", "")).strip().lower()
            if mk:
                rows_by_method[mk] = r

        methods_present = [m for m in want if m in rows_by_method]
        if not methods_present:
            continue

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
        n_seeds = [int(rows_by_method[m].get("n_seeds", 0) or 0) for m in methods_present]

        thr = add_solved_threshold_line(None, env_id)  # type: ignore[arg-type]
        y_vals = list(ci_lo) + list(ci_hi) + list(means)
        if thr is not None:
            y_vals.append(float(thr))

        y_mm = _finite_minmax(y_vals)
        if y_mm is None:
            continue

        x = np.arange(len(methods_present), dtype=np.float64)
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=int(DPI))

        yerr = np.vstack([np.maximum(0.0, means - ci_lo), np.maximum(0.0, ci_hi - means)])

        for i, mk in enumerate(methods_present):
            alpha = 1.0 if mk == "glpe" else 0.88
            z = 10 if mk == "glpe" else 2
            ax.bar(
                float(x[i]),
                float(means[i]),
                color=_color_for_method(mk),
                alpha=float(alpha),
                edgecolor="none",
                linewidth=0.0,
                zorder=z,
            )

        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            capthick=0.9,
            alpha=0.9,
            zorder=20,
        )

        span = max(1e-9, float(y_mm[1] - y_mm[0]))
        txt_off = 0.02 * span
        for xi, yi, n in zip(x.tolist(), means.tolist(), n_seeds):
            ax.text(
                float(xi),
                float(yi + txt_off) if yi >= 0.0 else float(yi - txt_off),
                f"n={int(n)}" if int(n) > 0 else "n=?",
                ha="center",
                va="bottom" if yi >= 0.0 else "top",
                fontsize=8,
                alpha=0.9,
                zorder=30,
            )

        labels = [str(label_by_key.get(m, m)) for m in methods_present]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Method")
        ax.set_ylabel("Mean episode return")
        ax.set_title(f"{env_id} — {title}")

        add_solved_threshold_line(ax, str(env_id))
        apply_grid(ax)
        _set_y_minmax(ax, float(y_mm[0]), float(y_mm[1]))

        fig.tight_layout()
        out = plots_root / f"{_env_tag(env_id)}__{filename_suffix}.png"
        save_fig_atomic(fig, out)
        plt.close(fig)
        written.append(out)

    return written
