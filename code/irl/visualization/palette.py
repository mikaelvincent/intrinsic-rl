from __future__ import annotations

import hashlib

_METHOD_COLORS: dict[str, str] = {
    "glpe": "#1B2A49",
    "vanilla": "#8EC7E8",
    "icm": "#93D18B",
    "rnd": "#B8A5D1",
    "ride": "#F3B07C",
    "riac": "#F08A8A",
    "glpe_lp_only": "#7ED6C2",
    "glpe_impact_only": "#9FA9E8",
    "glpe_nogate": "#D7C27A",
    "glpe_cache": "#D49AC6",
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

_COMPONENT_COLORS: dict[str, str] = {
    "env_step": _METHOD_COLORS["vanilla"],
    "policy": _METHOD_COLORS["icm"],
    "intrinsic": _METHOD_COLORS["glpe"],
    "gae": _METHOD_COLORS["ride"],
    "ppo": _METHOD_COLORS["riac"],
    "other": _METHOD_COLORS["rnd"],
    "reward": _METHOD_COLORS["vanilla"],
    "gate": _METHOD_COLORS["glpe"],
    "impact": _METHOD_COLORS["ride"],
    "lp": _METHOD_COLORS["glpe"],
    "learning_progress": _METHOD_COLORS["glpe"],
}


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


def normalize_component_key(component: object) -> str:
    return str(component).strip().lower()


def color_for_component(component: object, *, default: str | None = None) -> str:
    key = normalize_component_key(component)
    if key in _COMPONENT_COLORS:
        return _COMPONENT_COLORS[key]
    if default is not None:
        return str(default)
    return color_for_method(key)


def component_palette() -> dict[str, str]:
    return dict(_COMPONENT_COLORS)
