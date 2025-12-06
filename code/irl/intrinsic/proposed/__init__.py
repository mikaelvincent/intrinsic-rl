"""Proposed intrinsic module combining impact and learning progress.

Files
-----
* ``core``:
    Main :class:`Proposed` implementation.
* ``gating``:
    :class:`_RegionStats` container and region-gating update helpers.
* ``normalize``:
    Per-component RMS helpers for impact and learning-progress signals.

Public API
----------
``from irl.intrinsic.proposed import Proposed, _RegionStats``.
"""

from __future__ import annotations

import warnings

from .core import Proposed
from .gating import _RegionStats

__all__ = ["Proposed", "_RegionStats"]


def __getattr__(name: str):
    """Compatibility alias for legacy internal names.

    - `RegionStats` (old public-ish name) → `_RegionStats`
    """
    if name == "RegionStats":
        warnings.warn(
            "`RegionStats` has been renamed to `_RegionStats` in "
            "`irl.intrinsic.proposed`. Please update your imports.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _RegionStats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
