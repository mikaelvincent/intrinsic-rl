from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.style import GRID_ALPHA, DPI, FIG_WIDTH
from .plot_common import finite_quantiles, finite_std, get_result_by_name, pretty_name, run_meta_footer, save_fig, style


def _plot_cache_comparison(
    base: Mapping[str, Any],
    cached: Mapping[str, Any],
    out_path: Path,
    *,
    run_meta: Mapping[str, Any] | None = None,
) -> bool:
    qs_base = finite_quantiles(base.get("values"))
    qs_cached = finite_quantiles(cached.get("values"))
    std_base = finite_std(base.get("values"))
    std_cached = finite_std(cached.get("values"))
    if qs_base is None or qs_cached is None or std_base is None or std_cached is None:
        return False

    _b_q25, b_med, _b_q75 = qs_base
    _c_q25, c_med, _c_q75 = qs_cached

    if not (np.isfinite(b_med) and np.isfinite(c_med)):
        return False

    unit = str(base.get("unit", "")).strip() or str(cached.get("unit", "")).strip() or "transitions/s"

    base_ci = None
    cached_ci = None
    try:
        p = base.get("params")
        if isinstance(p, Mapping) and p.get("cache_interval") is not None:
            base_ci = int(p.get("cache_interval"))
    except Exception:
        base_ci = None
    try:
        p = cached.get("params")
        if isinstance(p, Mapping) and p.get("cache_interval") is not None:
            cached_ci = int(p.get("cache_interval"))
    except Exception:
        cached_ci = None

    base_label = "Cache off" if base_ci is None else f"Cache off (k={base_ci})"
    cached_label = "Cache on" if cached_ci is None else f"Cache on (k={cached_ci})"

    plt = style()
    fig, ax = plt.subplots(figsize=(float(FIG_WIDTH), 3.6), dpi=int(DPI))

    labels = [base_label, cached_label]
    x = np.arange(2, dtype=np.float64)

    medians = np.asarray([b_med, c_med], dtype=np.float64)
    stds = np.asarray([std_base, std_cached], dtype=np.float64)
    stds = np.where(np.isfinite(stds) & (stds >= 0.0), stds, 0.0)

    ax.bar(
        x,
        medians,
        color=[_color_for_method("glpe"), _color_for_method("glpe_cache")],
        alpha=0.9,
        edgecolor="none",
        linewidth=0.0,
        zorder=2,
    )
    ax.errorbar(
        x,
        medians,
        yerr=stds,
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=3,
        capthick=0.9,
        alpha=0.9,
        zorder=10,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(f"Throughput ({unit})")
    ax.grid(True, alpha=float(GRID_ALPHA))

    footer = run_meta_footer(run_meta)
    if footer:
        fig.text(0.01, 0.01, footer, ha="left", va="bottom", fontsize=8, alpha=0.9)
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    else:
        fig.tight_layout()

    save_fig(fig, Path(out_path))
    plt.close(fig)
    return True


def plot_speedup(
    results: list[Mapping[str, Any]], out_path: Path, *, run_meta: Mapping[str, Any] | None = None
) -> bool:
    base = get_result_by_name(results, "glpe.gate_median_cache.baseline")
    cached = get_result_by_name(results, "glpe.gate_median_cache.cached")
    if base is not None and cached is not None:
        if _plot_cache_comparison(base, cached, out_path, run_meta=run_meta):
            return True

    rows: list[dict[str, Any]] = []
    for r in results:
        if not isinstance(r, Mapping):
            continue
        if r.get("error"):
            continue
        unit = str(r.get("unit", "")).strip()
        metric = str(r.get("metric", "")).strip()
        if unit != "x" or metric != "speedup":
            continue

        qs = finite_quantiles(r.get("values"))
        if qs is None:
            continue
        _q25, med, _q75 = qs

        std = finite_std(r.get("values"))
        if std is None:
            continue

        name = str(r.get("name", "")).strip()
        if not name:
            continue
        rows.append(
            {
                "label": pretty_name(name),
                "median": float(med),
                "std": float(std),
            }
        )

    if not rows:
        return False

    plt = style()
    rows.sort(key=lambda x: float(x["median"]), reverse=True)

    labels = [r["label"] for r in rows]
    medians = np.asarray([float(r["median"]) for r in rows], dtype=np.float64)
    stds = np.asarray([float(r["std"]) for r in rows], dtype=np.float64)
    stds = np.where(np.isfinite(stds) & (stds >= 0.0), stds, 0.0)

    x = np.arange(len(rows), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(float(FIG_WIDTH), 3.6), dpi=int(DPI))

    ax.bar(x, medians, color=_color_for_method("glpe"), alpha=0.9, edgecolor="none", linewidth=0.0, zorder=2)
    ax.errorbar(
        x,
        medians,
        yerr=stds,
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=3,
        capthick=0.9,
        alpha=0.9,
        zorder=10,
    )
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Speedup (Ã—)")
    ax.grid(True, alpha=float(GRID_ALPHA))

    footer = run_meta_footer(run_meta)
    if footer:
        fig.text(0.01, 0.01, footer, ha="left", va="bottom", fontsize=8, alpha=0.9)
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    else:
        fig.tight_layout()

    save_fig(fig, Path(out_path))
    plt.close(fig)
    return True
