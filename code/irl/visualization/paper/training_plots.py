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
    draw_order,
    legend_order,
    linewidth_for_method,
    zorder_for_method,
)


def _env_tag(env_id: str) -> str:
    return str(env_id).replace("/", "-")


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


def _markevery(n: int, *, target_markers: int = 20) -> int:
    nn = int(max(1, n))
    tm = int(max(1, target_markers))
    return int(max(1, nn // tm))


def _merge_methods(by_method: Mapping[str, Sequence[Path]]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for m, dirs in by_method.items():
        mk = str(m).strip().lower()
        if not mk:
            continue
        bucket = out.setdefault(mk, [])
        if not isinstance(dirs, (list, tuple)):
            continue
        for p in dirs:
            try:
                bucket.append(Path(p))
            except Exception:
                continue
    return out


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

    align_mode = str(align).strip().lower() or "interpolate"
    if align_mode not in {"union", "intersection", "interpolate"}:
        raise ValueError("align must be one of: union, intersection, interpolate")

    plt = apply_rcparams_paper()
    written: list[Path] = []

    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: str(kv[0])):
        if not isinstance(by_method, Mapping):
            continue

        methods = _merge_methods(by_method)
        if not methods:
            continue

        series: dict[str, dict[str, object]] = {}

        for mk in draw_order(list(methods.keys())):
            run_dirs = methods.get(mk) or []
            if not run_dirs:
                continue

            agg_total = _aggregate_runs(
                run_dirs, metric="reward_total_mean", smooth=int(smooth), align=align_mode
            )
            agg_ext = _aggregate_runs(
                run_dirs, metric="reward_mean", smooth=int(smooth), align=align_mode
            )

            if int(getattr(agg_total, "n_runs", 0) or 0) <= 0:
                continue

            x = _as_f64_1d(getattr(agg_total, "steps", np.array([])))
            total_mean = _as_f64_1d(getattr(agg_total, "mean", np.array([])))

            if x.size == 0 or total_mean.size == 0:
                continue

            ext_mean = _interp(
                _as_f64_1d(getattr(agg_ext, "steps", np.array([]))),
                _as_f64_1d(getattr(agg_ext, "mean", np.array([]))),
                x,
            )

            n = int(min(x.size, total_mean.size, ext_mean.size))
            x = x[:n]
            total_mean = total_mean[:n]
            ext_mean = ext_mean[:n]

            finite = np.isfinite(x) & np.isfinite(total_mean) & np.isfinite(ext_mean)
            if not bool(finite.any()):
                continue

            x = x[finite]
            ext_mean = ext_mean[finite]
            intr_mean = (total_mean - ext_mean)[finite]

            series[str(mk)] = {
                "x": x,
                "extrinsic": ext_mean,
                "intrinsic": intr_mean,
                "n_runs": int(getattr(agg_total, "n_runs", 0) or 0),
            }

        if not series:
            continue

        fig, ax_ext = plt.subplots(figsize=FIGSIZE, dpi=int(DPI))
        ax_int = ax_ext.twinx()

        method_handles: dict[str, object] = {}
        method_labels: dict[str, str] = {}

        for mk in draw_order(list(series.keys())):
            rec = series.get(mk)
            if rec is None:
                continue

            x = np.asarray(rec["x"], dtype=np.float64).reshape(-1)
            y_ext = np.asarray(rec["extrinsic"], dtype=np.float64).reshape(-1)
            y_int = np.asarray(rec["intrinsic"], dtype=np.float64).reshape(-1)
            n_runs = int(rec.get("n_runs", 0) or 0)

            if x.size == 0 or y_ext.size == 0 or y_int.size == 0:
                continue

            c = _color_for_method(mk)
            lw = float(linewidth_for_method(mk))
            a = float(alpha_for_method(mk))
            z = int(zorder_for_method(mk))

            ln_ext = ax_ext.plot(
                x,
                y_ext,
                color=c,
                linewidth=lw,
                linestyle="-",
                alpha=a,
                zorder=z,
            )[0]

            me = _markevery(int(x.size), target_markers=20)
            ax_int.plot(
                x,
                y_int,
                color=c,
                linewidth=max(1.0, 0.9 * lw),
                linestyle="--",
                alpha=a,
                marker="o",
                markersize=3.0,
                markevery=me,
                markerfacecolor="none",
                markeredgecolor=c,
                markeredgewidth=0.9,
                zorder=z,
            )

            method_handles[mk] = ln_ext
            method_labels[mk] = f"{mk} (n={n_runs})"

        if not method_handles:
            plt.close(fig)
            continue

        ax_ext.set_xlabel("Environment steps")
        ax_ext.set_ylabel("Extrinsic reward (mean per step)")
        ax_int.set_ylabel("Intrinsic reward (mean per step)")

        ax_int.tick_params(axis="y", which="major", length=4, width=1)
        ax_int.tick_params(axis="y", which="minor", length=2, width=1)

        ax_ext.set_title(f"{env_id} — Reward decomposition (train)")
        apply_grid(ax_ext)

        ordered = [m for m in legend_order(list(method_handles.keys())) if m in method_handles]
        leg_methods = ax_ext.legend(
            [method_handles[m] for m in ordered],
            [method_labels[m] for m in ordered],
            loc="lower right",
            framealpha=float(LEGEND_FRAMEALPHA),
            fontsize=int(LEGEND_FONTSIZE),
        )
        ax_ext.add_artist(leg_methods)

        style_handles = [
            plt.Line2D([], [], color="black", linewidth=2.0, linestyle="-"),
            plt.Line2D(
                [],
                [],
                color="black",
                linewidth=2.0,
                linestyle="--",
                marker="o",
                markersize=3.0,
                markerfacecolor="none",
                markeredgewidth=0.9,
            ),
        ]
        ax_ext.legend(
            style_handles,
            ["Extrinsic (left axis)", "Intrinsic (right axis)"],
            loc="upper left",
            framealpha=float(LEGEND_FRAMEALPHA),
            fontsize=int(LEGEND_FONTSIZE),
        )

        out_path = out_root / f"{_env_tag(env_id)}__train_reward_decomp.png"
        fig.tight_layout()
        save_fig_atomic(fig, out_path)
        plt.close(fig)
        written.append(out_path)

    return written
