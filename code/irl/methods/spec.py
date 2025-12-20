from __future__ import annotations

from collections.abc import Sequence


def normalize_method(method: object) -> str:
    return str(method).strip().lower()


def canonical_method(method: object) -> str:
    m = normalize_method(method)
    if m.startswith("glpe_"):
        return "glpe"
    return m


def is_glpe_family(method: object) -> bool:
    return normalize_method(method).startswith("glpe")


_BASELINE_ORDER: tuple[str, ...] = ("vanilla", "icm", "rnd", "ride", "riac", "glpe")
_ABLATION_ORDER: tuple[str, ...] = (
    "glpe_lp_only",
    "glpe_impact_only",
    "glpe_nogate",
    "glpe_cache",
    "glpe",
)


def paper_method_groups(methods: Sequence[str]) -> tuple[list[str], list[str]]:
    ms: list[str] = []
    for m in methods:
        s = normalize_method(m)
        if s:
            ms.append(s)

    ms_set = set(ms)

    baselines: list[str] = [m for m in _BASELINE_ORDER if m in ms_set]
    extras = sorted(
        [m for m in ms if m not in set(baselines) and not m.startswith("glpe_") and m != "glpe"]
    )
    baselines.extend([m for m in extras if m not in set(baselines)])

    ablations: list[str] = [m for m in _ABLATION_ORDER if m in ms_set]
    other_abls = sorted([m for m in ms if m.startswith("glpe_") and m not in set(ablations)])
    ablations.extend([m for m in other_abls if m not in set(ablations)])

    if "glpe" in ms_set:
        if "glpe" not in baselines:
            baselines.append("glpe")
        if "glpe" not in ablations:
            ablations.append("glpe")

    return baselines, ablations


def suite_method_groups(all_methods: Sequence[str]) -> tuple[list[str], list[str]]:
    ms = [str(m) for m in all_methods]

    preferred = ["vanilla", "icm", "rnd", "ride", "riac"]
    baselines: list[str] = [m for m in preferred if m in ms]
    extras = [m for m in ms if m not in baselines and m != "glpe" and not str(m).startswith("glpe_")]
    baselines.extend(extras)
    if "glpe" in ms:
        baselines.append("glpe")

    ablation_priority = ["glpe_lp_only", "glpe_impact_only", "glpe_nogate", "glpe_cache"]
    ablations: list[str] = [m for m in ablation_priority if m in ms]
    other_abls = sorted([m for m in ms if str(m).startswith("glpe_") and m not in ablations])
    ablations.extend(other_abls)
    if "glpe" in ms:
        ablations.append("glpe")

    return baselines, ablations
