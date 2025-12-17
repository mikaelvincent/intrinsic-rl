from __future__ import annotations

from collections.abc import Sequence

from irl.visualization.suite_figures import (
    _generate_comparison_plot,
    _generate_component_plot,
    _generate_gating_plot,
    _generate_trajectory_plots,
)


def _suite_method_groups(all_methods: Sequence[str]) -> tuple[list[str], list[str]]:
    preferred = ["vanilla", "icm", "rnd", "ride", "riac"]
    baselines: list[str] = [m for m in preferred if m in all_methods]
    extras = [
        m
        for m in all_methods
        if m not in baselines and m != "glpe" and not str(m).startswith("glpe_")
    ]
    baselines.extend(extras)
    if "glpe" in all_methods:
        baselines.append("glpe")

    ablation_priority = ["glpe_lp_only", "glpe_impact_only", "glpe_nogate"]
    ablations: list[str] = [m for m in ablation_priority if m in all_methods]
    other_abls = sorted([m for m in all_methods if str(m).startswith("glpe_") and m not in ablations])
    ablations.extend(other_abls)
    if "glpe" in all_methods:
        ablations.append("glpe")

    return baselines, ablations


__all__ = [
    "_generate_comparison_plot",
    "_generate_component_plot",
    "_generate_gating_plot",
    "_generate_trajectory_plots",
    "_suite_method_groups",
]
