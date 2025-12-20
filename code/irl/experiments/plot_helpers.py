from __future__ import annotations

from collections.abc import Sequence

from irl.methods.spec import suite_method_groups as _suite_method_groups_impl
from irl.visualization.suite_figures import (
    _generate_comparison_plot,
    _generate_component_plot,
    _generate_gating_plot,
    _generate_trajectory_plots,
)


def _suite_method_groups(all_methods: Sequence[str]) -> tuple[list[str], list[str]]:
    return _suite_method_groups_impl(all_methods)


__all__ = [
    "_generate_comparison_plot",
    "_generate_component_plot",
    "_generate_gating_plot",
    "_generate_trajectory_plots",
    "_suite_method_groups",
]
