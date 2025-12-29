from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from irl.methods.spec import paper_method_groups as _paper_method_groups
from irl.visualization.data import aggregate_runs as _aggregate_runs
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.plot_utils import apply_rcparams_paper, save_fig_atomic, sort_env_ids as _sort_env_ids
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


def _has_glpe_and_variant(method_keys: Sequence[str]) -> bool:
    keys = {str(k).strip().lower() for k in method_keys if str(k).strip()}
    if "glpe" not in keys:
        return False
    return any(k.startswith("glpe_") for k in keys)


def _grid(n: int) -> tuple[int, int]:
    nn = int(n)
    if nn <= 0:
        return 0, 0
    ncols = 1 if nn <= 2 else 2
    nrows = int(math.ceil(float(nn) / float(ncols)))
    return nrows, ncols


def _figsize(nrows: int, ncols: int) -> tuple[float, float]:
    base_w, base_h = float(FIGSIZE[0]), float(FIGSIZE[1])
    w = base_w if int(ncols) <= 1 else base_w * 1.75
    h = base_h * float(max(1, int(nrows)))
    return float(w), float(h)


def _cleanup_training_reward_decomp(out_root: Path) -> None:
    out_root = Path(out_root)
    if not out_root.exists():
        return
    for pat in ("*__train_reward_decomp.png", "train_reward_decomp__*.png", "training_reward_decomp__*.png"):
        for p in out_root.glob(pat):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass


def _build_env_series(
    *,
    by_method: Mapping[str, Sequence[Path]],
    methods_to_plot: Sequence[str],
    smooth: int,
    align_mode: str,
) -> dict[str, dict[str, object]]:
    series: dict[str, dict[str, object]] = {}
    for mk in methods_to_plot:
        run_dirs = by_method.get(str(mk), None)
        if not run_dirs:
            continue

        agg_total = _aggregate_runs(
            list(run_dirs), metric="reward_total_mean", smooth=int(smooth), align=str(align_mode)
        )
        if int(getattr(agg_total, "n_runs", 0) or 0) <= 0:
            continue

        agg_ext = _aggregate_runs(
            list(run_dirs), metric="reward_mean", smooth=int(smooth), align=str(align_mode)
        )

        x = _as_f64_1d(getattr(agg_total, "steps", np.array([])))
        total_mean = _as_f64_1d(getattr(agg_total, "mean", np.array([])))

        ext_mean = _interp(
            _as_f64_1d(getattr(agg_ext, "steps", np.array([]))),
            _as_f64_1d(getattr(agg_ext, "mean", np.array([]))),
            x,
        )

        n = int(min(x.size, total_mean.size, ext_mean.size))
        if n <= 0:
            continue

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

    return series


def _plot_multienv_reward_decomp(
    series_by_env: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    title: str,
    out_path: Path,
) -> Path | None:
    envs = [str(e) for e in series_by_env.keys() if str(e).strip()]
    if not envs:
        return None

    nrows, ncols = _grid(len(envs))
    if nrows <= 0 or ncols <= 0:
        return None

    plt = apply_rcparams_paper()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=_figsize(nrows, ncols),
        dpi=int(DPI),
        squeeze=False,
    )
    axes_flat = list(axes.reshape(-1))

    methods_union: set[str] = set()

    for i, env_id in enumerate(envs):
        ax_ext = axes_flat[i]
        ax_int = ax_ext.twinx()

        series = series_by_env.get(env_id, {})
        if not series:
            ax_ext.axis("off")
            continue

        methods_union |= set(series.keys())

        col = int(i % ncols)
        row = int(i // ncols)

        for mk in draw_order(list(series.keys())):
            rec = series.get(mk)
            if rec is None:
                continue

            x = np.asarray(rec.get("x", np.array([])), dtype=np.float64).reshape(-1)
            y_ext = np.asarray(rec.get("extrinsic", np.array([])), dtype=np.float64).reshape(-1)
            y_int = np.asarray(rec.get("intrinsic", np.array([])), dtype=np.float64).reshape(-1)
            if x.size == 0 or y_ext.size == 0 or y_int.size == 0:
                continue

            n = int(min(x.size, y_ext.size, y_int.size))
            x = x[:n]
            y_ext = y_ext[:n]
            y_int = y_int[:n]

            c = _color_for_method(mk)
            lw = float(linewidth_for_method(mk))
            a = float(alpha_for_method(mk))
            z = int(zorder_for_method(mk))

            ax_ext.plot(
                x,
                y_ext,
                color=c,
                linewidth=lw,
                linestyle="-",
                alpha=a,
                zorder=z,
            )

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

        ax_ext.set_title(str(env_id))
        apply_grid(ax_ext)

        if row == nrows - 1:
            ax_ext.set_xlabel("Environment steps")

        if col == 0:
            ax_ext.set_ylabel("Extrinsic reward (mean per step)")

        if col == ncols - 1:
            ax_int.set_ylabel("Intrinsic reward (mean per step)")

        ax_int.tick_params(axis="y", which="major", length=4, width=1)
        ax_int.tick_params(axis="y", which="minor", length=2, width=1)

    for j in range(len(envs), len(axes_flat)):
        try:
            axes_flat[j].axis("off")
        except Exception:
            pass

    fig.suptitle(str(title))

    methods = legend_order(sorted(methods_union))
    method_handles = [
        plt.Line2D([], [], color=_color_for_method(m), lw=3.0 if str(m) == "glpe" else 2.0, linestyle="-")
        for m in methods
    ]
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

    bottom = 0.10 if method_handles else 0.06
    fig.tight_layout(rect=[0.0, bottom, 1.0, 0.94])

    if method_handles:
        fig.legend(
            handles=method_handles,
            labels=[str(m) for m in methods],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=int(min(6, max(1, len(method_handles)))),
            framealpha=float(LEGEND_FRAMEALPHA),
            fontsize=int(LEGEND_FONTSIZE),
        )

    fig.legend(
        handles=style_handles,
        labels=["Extrinsic (left axis)", "Intrinsic (right axis)"],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        framealpha=float(LEGEND_FRAMEALPHA),
        fontsize=int(LEGEND_FONTSIZE),
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig_atomic(fig, out_path)
    plt.close(fig)
    return out_path


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

    _cleanup_training_reward_decomp(out_root)

    env_methods: dict[str, dict[str, list[Path]]] = {}
    for env_id, by_method in groups_by_env.items():
        if not isinstance(by_method, Mapping):
            continue
        env_key = str(env_id).strip()
        if not env_key:
            continue
        merged = _merge_methods(by_method)
        if merged:
            env_methods[env_key] = merged

    if not env_methods:
        return []

    env_ids = _sort_env_ids(list(env_methods.keys()))

    all_methods: set[str] = set()
    for m in env_methods.values():
        all_methods |= set(m.keys())

    baselines, ablations = _paper_method_groups(sorted(all_methods))

    align_mode = str(align).strip().lower() or "interpolate"
    if align_mode not in {"union", "intersection", "interpolate"}:
        align_mode = "interpolate"

    outputs: list[Path] = []

    baseline_series_by_env: dict[str, dict[str, dict[str, object]]] = {}
    for env_id in env_ids:
        by_method = env_methods.get(env_id, {})
        want = [m for m in baselines if m in by_method]
        s = _build_env_series(by_method=by_method, methods_to_plot=want, smooth=int(smooth), align_mode=align_mode)
        if s:
            baseline_series_by_env[str(env_id)] = s

    if baseline_series_by_env:
        out_path = out_root / "train_reward_decomp__no_ablation.png"
        p = _plot_multienv_reward_decomp(
            baseline_series_by_env,
            title="Reward decomposition (train) - no ablation",
            out_path=out_path,
        )
        if p is not None:
            outputs.append(Path(p))

    ablation_series_by_env: dict[str, dict[str, dict[str, object]]] = {}
    for env_id in env_ids:
        by_method = env_methods.get(env_id, {})
        if not _has_glpe_and_variant(list(by_method.keys())):
            continue
        want = [m for m in ablations if m in by_method]
        s = _build_env_series(by_method=by_method, methods_to_plot=want, smooth=int(smooth), align_mode=align_mode)
        if s:
            ablation_series_by_env[str(env_id)] = s

    if ablation_series_by_env:
        out_path = out_root / "train_reward_decomp__ablation.png"
        p = _plot_multienv_reward_decomp(
            ablation_series_by_env,
            title="Reward decomposition (train) - ablation",
            out_path=out_path,
        )
        if p is not None:
            outputs.append(Path(p))

    return outputs
