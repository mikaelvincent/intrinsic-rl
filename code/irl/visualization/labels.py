from __future__ import annotations

import math
import re
from statistics import median
from typing import Iterable

from irl.visualization.style import LEGEND_FONTSIZE

_VERSION_RE = re.compile(r"-v\d+$", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)

_METHOD_LABELS: dict[str, str] = {
    "vanilla": "Vanilla",
    "icm": "ICM",
    "rnd": "RND",
    "ride": "RIDE",
    "riac": "RIAC",
    "glpe": "GLPE",
    "glpe_lp_only": "GLPE (LP only)",
    "glpe_impact_only": "GLPE (impact only)",
    "glpe_nogate": "GLPE (no gate)",
    "glpe_cache": "GLPE (cached)",
}

_COMPONENT_LABELS: dict[str, str] = {
    "env_step": "Environment step",
    "policy": "Policy",
    "intrinsic": "Intrinsic",
    "gae": "GAE",
    "ppo": "PPO",
    "other": "Other",
    "reward": "Extrinsic reward",
    "gate": "Gate rate",
    "impact": "Impact",
    "lp": "Learning progress",
}


def slugify(tag: object) -> str:
    s = str(tag).strip().lower()
    s = s.replace("paper_", "")
    s = _SLUG_RE.sub("-", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s or "plot"


def env_label(env_id: object) -> str:
    s = str(env_id).strip().replace("/", "-")
    s = _VERSION_RE.sub("", s)
    return s


def method_key(method: object) -> str:
    return str(method).strip().lower()


def method_label(method: object) -> str:
    k = method_key(method)
    if k in _METHOD_LABELS:
        return _METHOD_LABELS[k]
    if k.isupper():
        return k
    if len(k) <= 4 and k.isalpha():
        return k.upper()
    return k.replace("_", " ").strip()


def component_label(component: object) -> str:
    k = str(component).strip().lower()
    if k in _COMPONENT_LABELS:
        return _COMPONENT_LABELS[k]
    return k.replace("_", " ").strip().capitalize()


def legend_ncol(n_items: int, *, max_cols: int = 6) -> int:
    n = int(max(1, n_items))
    return int(min(max_cols, n))


def add_row_label(ax, label: str, *, x: float = -0.08, fontsize: int | None = None) -> None:
    fs = int(LEGEND_FONTSIZE) if fontsize is None else int(fontsize)
    ax.text(
        float(x),
        0.5,
        str(label),
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=fs,
        clip_on=False,
    )


def _median_subplot_row_gap(fig) -> float | None:
    axes = [ax for ax in getattr(fig, "axes", []) if getattr(ax, "get_visible", lambda: True)()]
    if len(axes) < 2:
        return None

    unique: dict[tuple[float, float, float, float], object] = {}
    for ax in axes:
        try:
            pos = ax.get_position()
        except Exception:
            continue
        x0, y0, x1, y1 = float(pos.x0), float(pos.y0), float(pos.x1), float(pos.y1)
        if not all(map(math.isfinite, (x0, y0, x1, y1))):
            continue
        if (x1 - x0) <= 1e-4 or (y1 - y0) <= 1e-4:
            continue
        key = (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))
        unique[key] = pos

    if len(unique) < 2:
        return None

    rows: dict[float, dict[str, float]] = {}
    for pos in unique.values():
        y_key = round(float(pos.y1), 3)
        rec = rows.get(y_key)
        if rec is None:
            rows[y_key] = {"top": float(pos.y1), "bottom": float(pos.y0)}
        else:
            rec["top"] = max(float(rec["top"]), float(pos.y1))
            rec["bottom"] = min(float(rec["bottom"]), float(pos.y0))

    if len(rows) < 2:
        return None

    row_list = sorted(rows.values(), key=lambda r: float(r["top"]), reverse=True)
    gaps: list[float] = []
    for upper, lower in zip(row_list[:-1], row_list[1:]):
        g = float(upper["bottom"]) - float(lower["top"])
        if math.isfinite(g) and g > 0.0:
            gaps.append(g)

    if not gaps:
        return None
    return float(median(gaps))


def add_legend_rows_top(
    fig,
    rows: Iterable[tuple[list[object], list[str], int]],
    *,
    y_top: float = 0.995,
    row_gap: float = 0.012,
    fontsize: int | None = None,
    pad_axes_pt: float | None = None,
    legend_kwargs: dict[str, object] | None = None,
) -> float:
    fs = int(LEGEND_FONTSIZE) if fontsize is None else int(fontsize)

    fig_h_in = 0.0
    try:
        fig_h_in = float(fig.get_figheight())
    except Exception:
        fig_h_in = 0.0
    fig_h_pt = 72.0 * fig_h_in if fig_h_in > 0.0 else 0.0

    def _pt_to_fig(pt: float) -> float:
        if fig_h_pt <= 0.0:
            return 0.0
        return float(pt) / float(fig_h_pt)

    max_row_gap_pt = 5.0
    gap_pt = float(row_gap) * float(fig_h_pt) if fig_h_pt > 0.0 else 0.0
    gap_pt = float(min(max_row_gap_pt, max(0.0, gap_pt)))
    gap_fig = _pt_to_fig(gap_pt)

    tight_layout_pad_mult = 1.08
    tight_pad_fig = _pt_to_fig(float(tight_layout_pad_mult) * float(fs))

    subplot_gap_fig: float | None = None
    try:
        fig.tight_layout()
        fig.canvas.draw()
        subplot_gap_fig = _median_subplot_row_gap(fig)
    except Exception:
        subplot_gap_fig = None

    if subplot_gap_fig is not None and math.isfinite(float(subplot_gap_fig)) and float(subplot_gap_fig) > 0.0:
        pad_axes_fig = float(subplot_gap_fig)
    else:
        if pad_axes_pt is None:
            pad_pt = float(0.006) * float(fig_h_pt) if fig_h_pt > 0.0 else 0.0
            pad_pt = float(min(5.0, max(2.5, pad_pt)))
        else:
            try:
                pad_pt = float(pad_axes_pt)
            except Exception:
                pad_pt = 0.0
            pad_pt = float(min(5.0, max(0.0, pad_pt)))
        pad_axes_fig = _pt_to_fig(pad_pt)

    y = float(y_top)
    legends: list[object] = []
    renderer = None

    base_leg_kwargs: dict[str, object] = {
        "frameon": False,
        "fontsize": fs,
        "handlelength": 2.2,
        "columnspacing": 1.2,
        "handletextpad": 0.6,
    }
    if isinstance(legend_kwargs, dict):
        for k, v in legend_kwargs.items():
            if v is None:
                continue
            base_leg_kwargs[str(k)] = v

    for handles, labels, ncol in rows:
        if not handles or not labels:
            continue

        leg = fig.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=int(max(1, ncol)),
            **base_leg_kwargs,
        )
        legends.append(leg)

        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = leg.get_window_extent(renderer=renderer)
            bbox_fig = bbox.transformed(fig.transFigure.inverted())
            y = float(bbox_fig.y0) - float(gap_fig)
        except Exception:
            y = float(y) - float(gap_fig)

    if not legends:
        return 1.0

    min_y0 = None
    try:
        if renderer is None:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

        bottoms: list[float] = []
        for leg in legends:
            bbox = leg.get_window_extent(renderer=renderer)
            bbox_fig = bbox.transformed(fig.transFigure.inverted())
            bottoms.append(float(bbox_fig.y0))

        if bottoms:
            min_y0 = float(min(bottoms))
    except Exception:
        min_y0 = None

    if min_y0 is None:
        min_y0 = float(y)

    top = float(min_y0) - float(pad_axes_fig) + float(tight_pad_fig)
    return float(max(0.0, min(1.0, top)))
