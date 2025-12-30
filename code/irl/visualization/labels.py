from __future__ import annotations

import re
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


def add_legend_rows_top(
    fig,
    rows: Iterable[tuple[list[object], list[str], int]],
    *,
    y_top: float = 0.995,
    row_gap: float = 0.045,
    fontsize: int | None = None,
) -> float:
    fs = int(LEGEND_FONTSIZE) if fontsize is None else int(fontsize)
    y = float(y_top)
    for handles, labels, ncol in rows:
        if not handles or not labels:
            continue
        fig.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            ncol=int(max(1, ncol)),
            frameon=False,
            fontsize=fs,
            handlelength=2.2,
            columnspacing=1.2,
            handletextpad=0.6,
        )
        y -= float(row_gap)
    return float(max(0.0, y - 0.01))
