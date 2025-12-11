"""Stats subpackage for Mann–Whitney U and bootstrap helpers.

The concrete implementations live in:

- :mod:`irl.stats.mannwhitney` for rank-based tests.
- :mod:`irl.stats.bootstrap` for generic bootstrap CIs.

Code elsewhere should continue to import from :mod:`irl.stats_utils`,
which re-exports this public surface.
"""

from __future__ import annotations

from .mannwhitney import Alt, MWUResult, rankdata, mannwhitney_u
from .bootstrap import bootstrap_ci

__all__ = [
    "Alt",
    "MWUResult",
    "rankdata",
    "mannwhitney_u",
    "bootstrap_ci",
]
