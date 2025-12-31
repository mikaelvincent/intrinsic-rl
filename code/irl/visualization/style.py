from __future__ import annotations

from typing import Any, Sequence

FIG_WIDTH: float = 7.0
FIGSIZE: tuple[float, float] = (FIG_WIDTH, 4.0)
DPI: int = 300

GRID_ALPHA: float = 0.25

LEGEND_FONTSIZE: int = 9
LEGEND_FRAMEALPHA: float = 0.0

PROPOSED_METHOD_KEY: str = "glpe"

# Canonical ordering for multi-panel figures and method legends.
_METHOD_ORDER: tuple[str, ...] = (
    "vanilla",
    "icm",
    "rnd",
    "ride",
    "riac",
    "glpe",
    "glpe_cache",
    "glpe_lp_only",
    "glpe_impact_only",
    "glpe_nogate",
)
_METHOD_RANK: dict[str, int] = {m: i for i, m in enumerate(_METHOD_ORDER)}

_ABLATION_DASH: dict[str, tuple[int, tuple[int, ...]]] = {
    "glpe_lp_only": (0, (6, 2)),
    "glpe_impact_only": (0, (3, 2)),
    "glpe_nogate": (0, (1, 2)),
    "glpe_cache": (0, (4, 2, 1, 2)),
}


def method_key(method: Any) -> str:
    return str(method).strip().lower()


def method_sort_key(method: Any) -> tuple[int, str]:
    k = method_key(method)
    if not k:
        return (10_000, "")
    return (int(_METHOD_RANK.get(k, 10_000)), k)


def order_methods(methods: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    uniq: list[str] = []
    for m in methods:
        k = method_key(m)
        if not k or k in seen:
            continue
        uniq.append(k)
        seen.add(k)
    uniq.sort(key=method_sort_key)
    return uniq


def is_proposed(method: Any) -> bool:
    return method_key(method) == PROPOSED_METHOD_KEY


def is_ablation(method: Any) -> bool:
    k = method_key(method)
    return k.startswith("glpe_") and k != PROPOSED_METHOD_KEY


def linestyle_for_method(method: Any) -> str | tuple[int, tuple[int, ...]]:
    k = method_key(method)
    if is_ablation(k):
        return _ABLATION_DASH.get(k, (0, (4, 2)))
    return "-"


def linewidth_for_method(method: Any) -> float:
    k = method_key(method)
    if k == PROPOSED_METHOD_KEY:
        return 1.8
    if is_ablation(k):
        return 0.9
    return 0.95


def alpha_for_method(method: Any) -> float:
    return 1.0 if is_proposed(method) else 0.88


def zorder_for_method(method: Any) -> int:
    k = method_key(method)
    if k == PROPOSED_METHOD_KEY:
        return 50
    r = _METHOD_RANK.get(k)
    if r is None:
        return 2
    return 2 + int(r)


def draw_order(methods: Sequence[str]) -> list[str]:
    ordered = order_methods(methods)
    if PROPOSED_METHOD_KEY in ordered:
        ordered = [m for m in ordered if m != PROPOSED_METHOD_KEY] + [PROPOSED_METHOD_KEY]
    return ordered


def legend_order(methods: Sequence[str]) -> list[str]:
    return order_methods(methods)


def apply_grid(ax) -> None:
    ax.grid(True, alpha=float(GRID_ALPHA))
