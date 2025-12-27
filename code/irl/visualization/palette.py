from __future__ import annotations

import hashlib


_METHOD_COLORS: dict[str, str] = {
    "vanilla": "#7f7f7f",
    "icm": "#ff7f0e",
    "rnd": "#1f77b4",
    "ride": "#8c564b",
    "riac": "#9467bd",
    "glpe": "#17becf",
    "glpe_cache": "#2ca02c",
    "glpe_lp_only": "#e377c2",
    "glpe_impact_only": "#bcbd22",
    "glpe_nogate": "#d62728",
}

_TAB20: tuple[str, ...] = (
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
)

_FALLBACK_COLORS: tuple[str, ...] = tuple(c for c in _TAB20 if c not in set(_METHOD_COLORS.values()))


def _stable_u32(*parts: str) -> int:
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


def normalize_method_key(method: object) -> str:
    return str(method).strip().lower()


def color_for_method(method: object, *, default: str | None = None) -> str:
    key = normalize_method_key(method)
    if key in _METHOD_COLORS:
        return _METHOD_COLORS[key]
    if default is not None:
        return str(default)
    if not _FALLBACK_COLORS:
        return "#000000"
    idx = _stable_u32("method_color", key) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]


def method_palette() -> dict[str, str]:
    return dict(_METHOD_COLORS)
