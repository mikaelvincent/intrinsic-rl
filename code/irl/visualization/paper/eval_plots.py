from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..palette import color_for_method as _color_for_method
from ..plot_utils import apply_rcparams_paper, save_fig_atomic


def _style():
    return apply_rcparams_paper()


def _save_fig(fig, path: Path) -> None:
    save_fig_atomic(fig, Path(path))


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
    if summary_df.empty:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    want = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
    if not want:
        return []

    label_by_key = (
        summary_df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .to_dict()
    )

    written: list[Path] = []
    plt = _style()

    for env_id in sorted(summary_df["env_id"].unique().tolist()):
        df_env = summary_df.loc[summary_df["env_id"] == env_id].copy()
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
            [float(rows_by_method[m]["mean_return_mean"]) for m in methods_present], dtype=np.float64
        )
        ci_lo = np.asarray(
            [float(rows_by_method[m]["mean_return_ci95_lo"]) for m in methods_present],
            dtype=np.float64,
        )
        ci_hi = np.asarray(
            [float(rows_by_method[m]["mean_return_ci95_hi"]) for m in methods_present],
            dtype=np.float64,
        )
        n_seeds = [int(rows_by_method[m].get("n_seeds", 0) or 0) for m in methods_present]

        y_minmax = _finite_minmax(list(ci_lo) + list(ci_hi) + list(means))
        if y_minmax is None:
            continue
        y_lo, y_hi = y_minmax

        labels = [str(label_by_key.get(m, m)) for m in methods_present]
        colors = [_color_for_method(m) for m in methods_present]

        x = np.arange(len(methods_present), dtype=np.float64)

        fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(methods_present)), 4.2))
        ax.bar(
            x,
            means,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )

        yerr = np.vstack([np.maximum(0.0, means - ci_lo), np.maximum(0.0, ci_hi - means)])
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
        )

        for xi, yi, n in zip(x.tolist(), means.tolist(), n_seeds):
            ax.text(
                float(xi),
                float(yi),
                f"n={int(n)}",
                ha="center",
                va="bottom" if yi >= 0.0 else "top",
                fontsize=8,
                alpha=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Method")
        ax.set_ylabel("Mean episode return (eval)")
        ax.set_title(f"{env_id} — {title}", loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        _set_y_minmax(ax, y_lo, y_hi)

        out = plots_root / f"{_env_tag(env_id)}__{filename_suffix}.png"
        _save_fig(fig, out)
        plt.close(fig)
        written.append(out)

    return written


def plot_eval_curves_by_env(
    by_step_df: pd.DataFrame,
    *,
    plots_root: Path,
    methods_to_plot: Sequence[str],
    title: str,
    filename_suffix: str,
) -> list[Path]:
    if by_step_df.empty:
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    want = [str(m).strip().lower() for m in methods_to_plot if str(m).strip()]
    if not want:
        return []

    label_by_key = (
        by_step_df.drop_duplicates(subset=["method_key"], keep="first")
        .set_index("method_key")["method"]
        .to_dict()
    )

    written: list[Path] = []
    plt = _style()

    for env_id in sorted(by_step_df["env_id"].unique().tolist()):
        df_env = by_step_df.loc[by_step_df["env_id"] == env_id].copy()
        if df_env.empty:
            continue

        methods_present = sorted(set(df_env["method_key"].tolist()) & set(want))
        if not methods_present:
            continue

        uniq_steps = sorted(set(df_env["ckpt_step"].tolist()))
        if len(uniq_steps) <= 1:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 4.6))

        all_ci_lo: list[float] = []
        all_ci_hi: list[float] = []

        for mk in want:
            if mk not in set(methods_present):
                continue

            df_m = df_env.loc[df_env["method_key"] == mk].copy()
            if df_m.empty:
                continue
            df_m = df_m.sort_values("ckpt_step")

            x = df_m["ckpt_step"].to_numpy(dtype=np.int64, copy=False)
            y = df_m["mean_return_mean"].to_numpy(dtype=np.float64, copy=False)
            lo = df_m["mean_return_ci95_lo"].to_numpy(dtype=np.float64, copy=False)
            hi = df_m["mean_return_ci95_hi"].to_numpy(dtype=np.float64, copy=False)

            all_ci_lo.extend(lo.tolist())
            all_ci_hi.extend(hi.tolist())

            c = _color_for_method(mk)
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=str(label_by_key.get(mk, mk)),
                color=c,
                alpha=0.95,
            )
            ax.fill_between(
                x,
                lo,
                hi,
                color=c,
                alpha=0.12,
                linewidth=0.0,
            )

        y_minmax = _finite_minmax(all_ci_lo + all_ci_hi)
        if y_minmax is not None:
            _set_y_minmax(ax, y_minmax[0], y_minmax[1])

        ax.set_xlabel("Checkpoint step (env steps)")
        ax.set_ylabel("Mean episode return (eval)")
        ax.set_title(f"{env_id} — {title}", loc="left", fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="best")

        out = plots_root / f"{_env_tag(env_id)}__{filename_suffix}.png"
        _save_fig(fig, out)
        plt.close(fig)
        written.append(out)

    return written
