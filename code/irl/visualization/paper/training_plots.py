from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from irl.visualization.data import aggregate_runs as _aggregate_runs
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic
from irl.visualization.style import (
    DPI,
    FIGSIZE,
    LEGEND_FRAMEALPHA,
    LEGEND_FONTSIZE,
    alpha_for_method,
    apply_grid,
    linestyle_for_method,
    linewidth_for_method,
    zorder_for_method,
)


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


def _method_tag(method: str) -> str:
    return str(method).strip().lower().replace("/", "-")


def _as_f64_1d(x: object) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _interp(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    if x_dst.size == 0:
        return np.empty((0,), dtype=np.float64)
    if x_src.size == 0 or y_src.size == 0:
        return np.full((int(x_dst.size),), np.nan, dtype=np.float64)

    n = int(min(x_src.size, y_src.size))
    xs = x_src[:n].astype(np.float64, copy=False)
    ys = y_src[:n].astype(np.float64, copy=False)

    if xs.size < 2:
        return np.full((int(x_dst.size),), float(ys[0]) if ys.size else np.nan, dtype=np.float64)

    return np.interp(x_dst, xs, ys).astype(np.float64, copy=False)


def plot_training_reward_decomposition(
    groups_by_env: Mapping[str, Mapping[str, Sequence[Path]]],
    *,
    plots_root: Path,
    smooth: int = 5,
    shade: bool = True,
    align: str = "interpolate",
) -> list[Path]:
    _ = shade
    if not isinstance(groups_by_env, Mapping):
        return []

    out_root = Path(plots_root) / "training"
    out_root.mkdir(parents=True, exist_ok=True)

    plt = apply_rcparams_paper()
    written: list[Path] = []

    align_mode = str(align).strip().lower() or "interpolate"
    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: str(kv[0])):
        if not isinstance(by_method, Mapping):
            continue

        for method, run_dirs in sorted(by_method.items(), key=lambda kv: str(kv[0])):
            if not isinstance(run_dirs, (list, tuple)) or not run_dirs:
                continue

            agg_total = _aggregate_runs(run_dirs, metric="reward_total_mean", smooth=int(smooth), align=align_mode)
            agg_ext = _aggregate_runs(run_dirs, metric="reward_mean", smooth=int(smooth), align=align_mode)

            if int(agg_total.n_runs) <= 0 or _as_f64_1d(agg_total.steps).size == 0:
                continue
            if int(agg_ext.n_runs) <= 0 or _as_f64_1d(agg_ext.steps).size == 0:
                continue

            x = _as_f64_1d(agg_total.steps)
            total_mean = _as_f64_1d(agg_total.mean)

            ext_mean = _interp(_as_f64_1d(agg_ext.steps), _as_f64_1d(agg_ext.mean), x)

            agg_int = _aggregate_runs(run_dirs, metric="r_int_mean", smooth=int(smooth), align=align_mode)
            has_int = int(agg_int.n_runs) > 0 and _as_f64_1d(agg_int.steps).size > 0
            if has_int:
                int_mean = _interp(_as_f64_1d(agg_int.steps), _as_f64_1d(agg_int.mean), x)
            else:
                int_mean = total_mean - ext_mean

            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=int(DPI))

            ax.plot(
                x,
                ext_mean,
                label="reward_mean (extrinsic)",
                color=_color_for_method("vanilla"),
                linewidth=1.9,
                alpha=0.88,
                zorder=2,
            )

            ax.plot(
                x,
                int_mean,
                label="r_int_mean (intrinsic)",
                color=_color_for_method(str(method)),
                linestyle=(0, (3, 2)),
                linewidth=1.8,
                alpha=0.88,
                zorder=3,
            )

            mk = str(method).strip().lower()
            ax.plot(
                x,
                total_mean,
                label="reward_total_mean (total)",
                color=_color_for_method(mk),
                linewidth=float(linewidth_for_method(mk)),
                linestyle=linestyle_for_method(mk),
                alpha=float(alpha_for_method(mk)),
                zorder=int(zorder_for_method(mk)),
            )

            ax.set_xlabel("Environment steps")
            ax.set_ylabel("Reward (mean per step)")
            ax.set_title(f"{env_id} — {method} reward decomposition (train)")

            apply_grid(ax)
            ax.legend(loc="lower right", framealpha=float(LEGEND_FRAMEALPHA), fontsize=int(LEGEND_FONTSIZE))

            out_path = out_root / f"{_env_tag(env_id)}__{_method_tag(method)}__train_reward_decomp.png"
            fig.tight_layout()
            save_fig_atomic(fig, out_path)
            plt.close(fig)
            written.append(out_path)

    return written
