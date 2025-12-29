from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import typer

from irl.visualization.data import aggregate_runs
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
    linestyle_for_method,
    linewidth_for_method,
    zorder_for_method,
)


def _is_effectively_one(vals: np.ndarray, *, tol: float) -> bool:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return True
    return float(np.max(np.abs(arr - 1.0))) <= float(tol)


def _method_order(methods: list[str]) -> list[str]:
    order = [
        "vanilla",
        "icm",
        "rnd",
        "ride",
        "riac",
        "glpe_lp_only",
        "glpe_impact_only",
        "glpe_nogate",
        "glpe_cache",
        "glpe",
    ]
    idx = {m: i for i, m in enumerate(order)}

    def key(m: str) -> tuple[int, str]:
        ml = str(m).strip().lower()
        if ml in idx:
            return idx[ml], ml
        if ml.startswith("glpe_"):
            return 90, ml
        return 100, ml

    return sorted(list(methods), key=key)


def plot_intrinsic_taper_weight(
    groups_by_env: Mapping[str, Mapping[str, list[Path]]],
    *,
    plots_root: Path,
    smooth: int = 1,
    shade: bool = True,
    align: str = "interpolate",
    inactive_tol: float = 1e-6,
) -> list[Path]:
    _ = shade
    if not isinstance(groups_by_env, Mapping):
        return []

    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    align_mode = str(align).strip().lower() or "interpolate"
    if align_mode not in {"union", "intersection", "interpolate"}:
        raise ValueError("align must be one of: union, intersection, interpolate")

    plt = apply_rcparams_paper()

    written: list[Path] = []

    for env_id, by_method in sorted(groups_by_env.items(), key=lambda kv: str(kv[0])):
        if not isinstance(by_method, Mapping):
            continue

        glpe_methods: dict[str, list[Path]] = {}
        for m, dirs in by_method.items():
            ml = str(m).strip().lower()
            if ml.startswith("glpe") and isinstance(dirs, (list, tuple)) and dirs:
                glpe_methods[ml] = [Path(p) for p in dirs]

        if not glpe_methods:
            continue

        aggs: list[tuple[str, object]] = []
        any_active = False

        for m in _method_order(list(glpe_methods.keys())):
            dirs = glpe_methods.get(m, [])
            if not dirs:
                continue

            try:
                agg = aggregate_runs(dirs, metric="intrinsic_taper_weight", smooth=int(smooth), align=align_mode)
            except Exception:
                continue

            if getattr(agg, "n_runs", 0) <= 0 or getattr(agg, "steps", np.array([])).size == 0:
                continue

            mean_vals = np.asarray(getattr(agg, "mean"), dtype=np.float64)
            if not _is_effectively_one(mean_vals, tol=float(inactive_tol)):
                any_active = True

            aggs.append((m, agg))

        if not aggs or not any_active:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=int(DPI))

        for m in draw_order([mm for mm, _ in aggs]):
            agg = next((a for k, a in aggs if k == m), None)
            if agg is None:
                continue

            steps = np.asarray(getattr(agg, "steps"), dtype=np.int64)
            mean = np.asarray(getattr(agg, "mean"), dtype=np.float64)

            if steps.size == 0 or mean.size == 0:
                continue

            ax.plot(
                steps,
                mean,
                label=f"{m} (n={int(getattr(agg, 'n_runs', 0) or 0)})",
                linewidth=float(linewidth_for_method(m)),
                linestyle=linestyle_for_method(m),
                alpha=float(alpha_for_method(m)),
                color=_color_for_method(m),
                zorder=int(zorder_for_method(m)),
            )

        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Intrinsic taper weight")
        ax.set_title(f"{env_id} — GLPE intrinsic taper weight")
        ax.set_ylim(-0.05, 1.05)

        apply_grid(ax)
        ax.legend(loc="lower right", framealpha=float(LEGEND_FRAMEALPHA), fontsize=int(LEGEND_FONTSIZE))

        env_tag = str(env_id).replace("/", "-")
        out_path = plots_root / f"{env_tag}__glpe_intrinsic_taper.png"
        fig.tight_layout()
        save_fig_atomic(fig, out_path)
        plt.close(fig)

        written.append(out_path)
        typer.echo(f"[suite] Saved taper plot: {out_path}")

    return written
