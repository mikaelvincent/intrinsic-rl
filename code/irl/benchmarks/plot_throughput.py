from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from irl.visualization.palette import color_for_component as _color_for_component
from irl.visualization.palette import color_for_method as _color_for_method
from irl.visualization.style import GRID_ALPHA, DPI, FIG_WIDTH
from .plot_common import finite_quantiles, finite_std, pretty_name, save_fig, style


def _bench_color(name: str) -> str:
    s = str(name).strip().lower()

    if s.startswith("glpe.gate_median_cache.cached"):
        return _color_for_method("glpe_cache")
    if s.startswith("glpe."):
        return _color_for_method("glpe")
    if s.startswith("riac."):
        return _color_for_method("riac")
    if s.startswith("ppo."):
        return _color_for_component("ppo")
    if s.startswith("gae."):
        return _color_for_component("gae")
    if s.startswith("env."):
        return _color_for_component("env_step")
    if s.startswith("kdtree."):
        return _color_for_component("other")
    return _color_for_component("other")


def plot_throughput(
    results: list[Mapping[str, Any]], out_path: Path, *, run_meta: Mapping[str, Any] | None = None
) -> bool:
    _ = run_meta
    rows_by_unit: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if not isinstance(r, Mapping):
            continue
        if r.get("error"):
            continue
        unit = str(r.get("unit", "")).strip()
        if not unit.endswith("/s"):
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
        rows_by_unit.setdefault(unit, []).append(
            {
                "name": name,
                "label": pretty_name(name),
                "median": float(med),
                "std": float(std),
                "color": _bench_color(name),
            }
        )

    if not rows_by_unit:
        return False

    plt = style()
    preferred_units = ["transitions/s", "samples/s", "points/s"]
    units = [u for u in preferred_units if u in rows_by_unit] + sorted(
        [u for u in rows_by_unit.keys() if u not in set(preferred_units)]
    )

    n_units = len(units)
    total_rows = sum(len(rows_by_unit[u]) for u in units)
    height = max(2.8, 1.0 * n_units + 0.32 * total_rows)
    fig, axes = plt.subplots(
        n_units, 1, figsize=(float(FIG_WIDTH), float(height)), squeeze=False, dpi=int(DPI)
    )

    for ax, unit in zip(axes[:, 0], units):
        rows = list(rows_by_unit[unit])
        rows.sort(key=lambda x: float(x["median"]), reverse=True)

        labels = [r["label"] for r in rows]
        medians = np.asarray([float(r["median"]) for r in rows], dtype=np.float64)
        stds = np.asarray([float(r["std"]) for r in rows], dtype=np.float64)
        stds = np.where(np.isfinite(stds) & (stds >= 0.0), stds, 0.0)

        y = np.arange(len(rows), dtype=np.float64)
        colors = [str(r.get("color", _color_for_component("other"))) for r in rows]

        ax.barh(y, medians, color=colors, alpha=0.9, edgecolor="none", linewidth=0.0, zorder=2)
        ax.errorbar(
            medians,
            y,
            xerr=stds,
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            capthick=0.9,
            alpha=0.9,
            zorder=10,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"Throughput ({unit})")
        ax.grid(True, alpha=float(GRID_ALPHA))

        finite_pos = medians[np.isfinite(medians) & (medians > 0.0)]
        if finite_pos.size >= 2:
            ratio = float(finite_pos.max() / finite_pos.min())
            if ratio >= 50.0:
                ax.set_xscale("log")

    fig.tight_layout()

    save_fig(fig, Path(out_path))
    plt.close(fig)
    return True
