"""Statistical utilities: Mann–Whitney U and simple bootstrapping.

This module remains the public facade. The concrete implementations now
live under :mod:`irl.stats`:

- :mod:`irl.stats.mannwhitney` for rank-based tests.
- :mod:`irl.stats.bootstrap` for bootstrap confidence intervals.
"""

from __future__ import annotations

from irl.stats.mannwhitney import Alt, MWUResult, rankdata, mannwhitney_u
from irl.stats.bootstrap import bootstrap_ci

__all__ = [
    "Alt",
    "MWUResult",
    "rankdata",
    "mannwhitney_u",
    "bootstrap_ci",
]
